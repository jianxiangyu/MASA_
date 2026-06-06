set -x

MODEL_SIZE="4b"                    
SKILL_JSON="memory_data/webshop/4b_masa_skills.json"

case "$MODEL_SIZE" in
    4b)
        MODEL_PATH="Qwen3-4B"
        TP_SIZE=1; GPU_MEM=0.6; MICRO_BATCH=8; LOG_PROB_BATCH=16; REF_BATCH=8; MAX_SEQS=256 ;;
    8b)
        MODEL_PATH="Qwen3-8B"
        TP_SIZE=1; GPU_MEM=0.6; MICRO_BATCH=8; LOG_PROB_BATCH=16; REF_BATCH=8; MAX_SEQS=256 ;;
    14b)
        MODEL_PATH="Qwen3-14B"
        TP_SIZE=2; GPU_MEM=0.55; MICRO_BATCH=4; LOG_PROB_BATCH=8; REF_BATCH=4; MAX_SEQS=128 ;;
    32b)
        MODEL_PATH="Qwen3-32B"
        TP_SIZE=4; GPU_MEM=0.5; MICRO_BATCH=4; LOG_PROB_BATCH=4; REF_BATCH=4; MAX_SEQS=128 ;;
esac

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export PATH=$JAVA_HOME/bin:$PATH

EMBEDDING_MODEL="Qwen3-Embedding-0.6B"
num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=500
group_size=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/verl-agent/text/train.parquet \
    data.val_files=./data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_BATCH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_SEQS \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=$SKILL_JSON \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=$EMBEDDING_MODEL \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.task_specific_top_k=5 \
    +env.skills_only_memory.enable_dynamic_update=False \
    +env.skills_only_memory.update_threshold=0.4 \
    +env.skills_only_memory.max_new_skills=3 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name="webshop_qwen3_${MODEL_SIZE}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True
