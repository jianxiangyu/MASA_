# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        # Add retrieval memory or skills-only memory if configured
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieval_memory = SkillsOnlyMemory(
                skills_json_path=som_cfg.skills_json_path,
                retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                embedding_model_path=som_cfg.get('embedding_model_path', None),
                task_specific_top_k=som_cfg.get('task_specific_top_k', None),
            )
            self.retrieved_memories = None
            print(f"[SearchEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                skills_json_path=config.env.retrieval_memory.get('skills_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[SearchEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None
            self.retrieved_memories = None

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        self.kwargs = kwargs
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))
        if self.retrieval_memory is not None:
            self.retrieved_memories = []

            # Determine which config to use
            if self.config.env.get('use_skills_only_memory', False):
                mem_config = self.config.env.skills_only_memory
            else:
                mem_config = self.config.env.retrieval_memory

            for task in self.tasks:
                memories = self.retrieval_memory.retrieve(
                    task_description=task,
                    top_k=mem_config.get('top_k', 10),
                    similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                    max_tokens=mem_config.get('max_tokens', 2000),
                    include_examples=mem_config.get('include_examples', False)
                )
                self.retrieved_memories.append(memories)

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            # Use retrieval memory template if enabled
            has_retrieval = (self.retrieval_memory is not None and
                           self.retrieved_memories is not None)
            if init or self.config.env.history_length <= 0:
                if has_retrieval:
                    # First step WITH skills injection
                    memory_context = self.retrieval_memory.format_for_prompt(
                        self.retrieved_memories[i]
                    )
                    obs_i = SEARCH_TEMPLATE_WITH_MEMORY.format(
                        task_description=self.tasks[i],
                        retrieved_memories=memory_context,
                        step_count=0,
                        memory_context="None",
                    )
                else:
                    obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i]
                    )
            elif has_retrieval:
                # Format retrieved memories for prompt
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i]
                )
                obs_i = SEARCH_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    memory_context=memory_ctx[i],
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()

        # Add retrieval memory or skills-only memory if configured
        if config.env.get('use_skills_only_memory', False):
            som_cfg = config.env.skills_only_memory
            memory_class = som_cfg.get('memory_class', 'default')

            if memory_class == 'ablation':
                from agent_system.memory import AblationSkillsMemory
                # Convert OmegaConf list to plain list if needed
                atom_subset = som_cfg.get('atom_subset', None)
                if atom_subset is not None:
                    atom_subset = list(atom_subset)
                self.retrieval_memory = AblationSkillsMemory(
                    skills_json_path=som_cfg.skills_json_path,
                    retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                    embedding_model_path=som_cfg.get('embedding_model_path', None),
                    task_specific_top_k=som_cfg.get('task_specific_top_k', None),
                    atom_subset=atom_subset,
                    ablation_preset=som_cfg.get('ablation_preset', None),
                )
            else:
                from agent_system.memory import SkillsOnlyMemory
                self.retrieval_memory = SkillsOnlyMemory(
                    skills_json_path=som_cfg.skills_json_path,
                    retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                    embedding_model_path=som_cfg.get('embedding_model_path', None),
                    task_specific_top_k=som_cfg.get('task_specific_top_k', None),
                )
            self.retrieved_memories = None
            print(f"[AlfWorldEnvironmentManager] Skills-only memory enabled "
                  f"(class={memory_class}, mode={som_cfg.get('retrieval_mode', 'template')})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                skills_json_path=config.env.retrieval_memory.get('skills_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[AlfWorldEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        # Retrieve memories for each task if enabled
        if self.retrieval_memory is not None:
            self.retrieved_memories = []

            # Determine which config to use
            if self.config.env.get('use_skills_only_memory', False):
                mem_config = self.config.env.skills_only_memory
            else:
                mem_config = self.config.env.retrieval_memory

            for task in self.tasks:
                memories = self.retrieval_memory.retrieve(
                    task_description=task,
                    top_k=mem_config.get('top_k', 10),
                    similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                    max_tokens=mem_config.get('max_tokens', 2000),
                    include_examples=mem_config.get('include_examples', False)
                )
                self.retrieved_memories.append(memories)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            # Check if retrieval memory is available
            has_retrieval = (self.retrieval_memory is not None and
                           self.retrieved_memories is not None)

            if init or self.config.env.history_length <= 0:
                if has_retrieval:
                    # First step WITH skills injection
                    memory_context = self.retrieval_memory.format_for_prompt(
                        self.retrieved_memories[i]
                    )
                    obs = ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                        task_description=self.tasks[i],
                        retrieved_memories=memory_context,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
                else:
                    obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
            elif has_retrieval:
                # Format retrieved memories for prompt
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i]
                )
                obs = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    def save_episode_trajectories(self, batch_data_list, infos_list):
        """
        Save successful/failed trajectories from completed episodes to memory pool.

        Args:
            batch_idx: Index of the batch
            total_batch_list: List of batch data containing trajectories
            infos: List of info dicts containing episode metadata
        """
        if self.retrieval_memory is None:
            return

        save_dir = self.config.env.retrieval_memory.get('save_dir', None)
        if save_dir is None:
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'new_memories.json')

        # Iterate through each environment
        for env_idx in range(len(self.tasks)):
            # Check if episode is done
            # We'll save trajectories when episodes complete
            # This will be called from the trainer after validation/training episodes
            pass  # Actual saving logic will be called from trainer


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()

        # Skills-only memory (same interface as AlfWorldEnvironmentManager)
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieval_memory = SkillsOnlyMemory(
                skills_json_path=som_cfg.skills_json_path,
                retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                embedding_model_path=som_cfg.get('embedding_model_path', None),
                task_specific_top_k=som_cfg.get('task_specific_top_k', None),
            )
            self.retrieved_memories = None
            print(f"[WebshopEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)

        # Retrieve skills for each task if memory is configured
        if self.retrieval_memory is not None:
            mem_cfg = self.config.env.skills_only_memory
            self.retrieved_memories = [
                self.retrieval_memory.retrieve(
                    task_description=task,
                    top_k=mem_cfg.get('top_k', 6),
                )
                for task in self.tasks
            ]

        observations = {'text': self.build_text_obs(obs, infos, init=True),
                        'image': None,
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        has_retrieval = (
            self.retrieval_memory is not None
            and self.retrieved_memories is not None
        )

        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                if has_retrieval:
                    memory_context = self.retrieval_memory.format_for_prompt(
                        self.retrieved_memories[i]
                    )
                    obs = WEBSHOP_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                        task_description=self.tasks[i],
                        retrieved_memories=memory_context,
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
                else:
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
            elif has_retrieval:
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i]
                )
                obs = WEBSHOP_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            if len(obs) > 13000:
                print(f"Warning len(obs)={len(obs)} is too long")
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _detect_category(self, task_description: str) -> str:
        """Detect product category from task description for per-category metrics."""
        goal = task_description.lower()
        if any(kw in goal for kw in [
            'shirt', 'dress', 'jacket', 'pant', 'coat', 'sweater',
            'blouse', 'clothing', 'clothes', 't-shirt', 'hoodie', 'vest',
            'shorts', 'jeans', 'skirt', 'suit',
        ]):
            return 'apparel'
        elif any(kw in goal for kw in [
            'shoe', 'boot', 'sneaker', 'sandal', 'heel', 'slipper',
            'footwear', 'loafer',
        ]):
            return 'footwear'
        elif any(kw in goal for kw in [
            'laptop', 'phone', 'computer', 'tablet', 'charger',
            'cable', 'headphone', 'speaker', 'camera', 'electronic',
            'battery', 'adapter', 'usb', 'bluetooth',
        ]):
            return 'electronics'
        elif any(kw in goal for kw in [
            'necklace', 'ring', 'bracelet', 'earring', 'watch',
            'jewelry', 'bag', 'purse', 'wallet', 'handbag',
        ]):
            return 'accessories'
        elif any(kw in goal for kw in [
            'furniture', 'lamp', 'curtain', 'pillow', 'bedding',
            'decor', 'candle', 'vase', 'rug', 'shelf',
        ]):
            return 'home_decor'
        elif any(kw in goal for kw in [
            'cream', 'lotion', 'shampoo', 'conditioner', 'moisturizer',
            'serum', 'makeup', 'beauty', 'vitamin', 'supplement',
            'soap', 'perfume', 'fragrance', 'sunscreen',
        ]):
            return 'beauty_health'
        else:
            return 'other'

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)

                # Per-category metrics
                category = self._detect_category(self.tasks[batch_idx])
                cat_sr_key = f'{category}_success_rate'
                cat_ts_key = f'{category}_task_score'
                if cat_sr_key not in success:
                    success[cat_sr_key] = []
                if cat_ts_key not in success:
                    success[cat_ts_key] = []
                success[cat_sr_key].append(won_value)
                success[cat_ts_key].append(score_value)
                return

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)