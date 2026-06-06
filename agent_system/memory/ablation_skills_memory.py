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

"""
Ablation-aware skills memory for Pilot Ablation Experiments.

Extends SkillsOnlyMemory to support 6-atom skill format and selective atom
assembly via ``atom_subset`` parameter.

The 6 atoms are:
  - title:       Skill name
  - strategy:    High-level approach summary
  - steps:       Detailed step-by-step procedure (list)
  - constraints: Rules / prohibitions / warnings (list)
  - example:     Concrete worked example
  - trigger:     When to apply this skill

Ablation combinations:
  full          = all 6 atoms
  -title        = all except title
  -strategy     = all except strategy
  -steps        = all except steps
  -constraints  = all except constraints
  -example      = all except example
  -trigger      = all except trigger
  minimal       = title only
  no_skill      = empty (use noskill.json or empty list)
"""

import json
import os
from typing import Dict, Any, List, Optional, Set
from .skills_only_memory import SkillsOnlyMemory


# All 6 atoms
ALL_ATOMS = frozenset({"title", "strategy", "steps", "constraints", "example", "trigger"})

# Pre-defined ablation combinations
ABLATION_PRESETS: Dict[str, Set[str]] = {
    "full":         set(ALL_ATOMS),
    "-title":       ALL_ATOMS - {"title"},
    "-strategy":    ALL_ATOMS - {"strategy"},
    "-steps":       ALL_ATOMS - {"steps"},
    "-constraints": ALL_ATOMS - {"constraints"},
    "-example":     ALL_ATOMS - {"example"},
    "-trigger":     ALL_ATOMS - {"trigger"},
    "minimal":      {"title"},
    "no_skill":     set(),  # nothing
}


class AblationSkillsMemory(SkillsOnlyMemory):
    """
    Skills memory with ablation support for 6-atom skill format.

    Inherits all retrieval logic from :class:`SkillsOnlyMemory` and overrides
    :meth:`format_for_prompt` to assemble skill text using only the specified
    atom subset.

    Usage via Hydra config::

        +env.use_skills_only_memory=True
        +env.skills_only_memory.memory_class=ablation
        +env.skills_only_memory.skills_json_path=memory_data/alfworld/verbose_skills_6atoms.json
        +env.skills_only_memory.atom_subset=[title,strategy,steps,constraints,example,trigger]
        # or use preset:
        +env.skills_only_memory.ablation_preset=full
    """

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
        atom_subset: Optional[List[str]] = None,
        ablation_preset: Optional[str] = None,
    ):
        """
        Args:
            skills_json_path:     Path to 6-atom skills JSON file.
            retrieval_mode:       ``"template"`` or ``"embedding"``.
            embedding_model_path: Embedding model path (for embedding mode).
            task_specific_top_k:  Max task-specific skills to return.
            atom_subset:          Explicit list of atoms to include, e.g.
                                  ``["title", "strategy", "steps"]``.
            ablation_preset:      Name of a pre-defined ablation combination
                                  (e.g. ``"full"``, ``"-steps"``, ``"minimal"``).
                                  Overridden by ``atom_subset`` if both given.
        """
        # Initialize parent (loads skills, sets up retrieval)
        super().__init__(
            skills_json_path=skills_json_path,
            retrieval_mode=retrieval_mode,
            embedding_model_path=embedding_model_path,
            task_specific_top_k=task_specific_top_k,
        )

        # Determine active atom set
        if atom_subset is not None:
            self.atom_subset = set(atom_subset)
        elif ablation_preset is not None:
            if ablation_preset not in ABLATION_PRESETS:
                raise ValueError(
                    f"Unknown ablation_preset '{ablation_preset}'. "
                    f"Available: {list(ABLATION_PRESETS.keys())}"
                )
            self.atom_subset = ABLATION_PRESETS[ablation_preset]
        else:
            # Default: full (all atoms)
            self.atom_subset = set(ALL_ATOMS)

        # Validate atoms
        unknown = self.atom_subset - ALL_ATOMS
        if unknown:
            raise ValueError(f"Unknown atoms: {unknown}. Valid: {ALL_ATOMS}")

        preset_name = ablation_preset or "custom"
        for name, atoms in ABLATION_PRESETS.items():
            if self.atom_subset == atoms:
                preset_name = name
                break

        print(
            f"[AblationSkillsMemory] Active atoms: {sorted(self.atom_subset)} "
            f"(preset={preset_name})"
        )

    # ------------------------------------------------------------------ #
    # Skill text assembly                                                  #
    # ------------------------------------------------------------------ #

    def _assemble_skill_text(self, skill: Dict[str, Any]) -> str:
        """
        Assemble a single skill dict into a formatted text string using only
        the active atom subset.

        Args:
            skill: A skill dict with 6-atom fields.

        Returns:
            Formatted multi-line string for this skill.
        """
        parts: List[str] = []
        atoms = self.atom_subset

        # Title — always a heading if present
        if "title" in atoms:
            title = skill.get("title", "")
            if title:
                parts.append(f"**{title}**")

        # Strategy — one-line summary
        if "strategy" in atoms:
            strategy = skill.get("strategy", "")
            if strategy:
                parts.append(f"Strategy: {strategy}")

        # Steps — numbered list
        if "steps" in atoms:
            steps = skill.get("steps", [])
            if isinstance(steps, list) and steps:
                parts.append("Steps:")
                for step in steps:
                    parts.append(f"  {step}")
            elif isinstance(steps, str) and steps:
                parts.append(f"Steps: {steps}")

        # Constraints — bullet list
        if "constraints" in atoms:
            constraints = skill.get("constraints", [])
            if isinstance(constraints, list) and constraints:
                parts.append("Constraints:")
                for c in constraints:
                    parts.append(f"  - {c}")
            elif isinstance(constraints, str) and constraints:
                parts.append(f"Constraints: {constraints}")

        # Example — concrete illustration
        if "example" in atoms:
            example = skill.get("example", "")
            if example:
                parts.append(f"Example: {example}")

        # Trigger — when to apply
        if "trigger" in atoms:
            trigger = skill.get("trigger", "")
            if trigger:
                parts.append(f"When to apply: {trigger}")

        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Override: format_for_prompt                                          #
    # ------------------------------------------------------------------ #

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """
        Format retrieved skills into a prompt-ready string using only the
        active atom subset.

        This overrides the parent's ``format_for_prompt`` to support 6-atom
        skill dicts instead of the old ``principle`` / ``when_to_apply`` format.

        Args:
            retrieved_memories: Dict returned by :meth:`retrieve`.

        Returns:
            Formatted multi-section string to insert into the agent prompt.
        """
        # If no atoms selected (no_skill preset), return empty
        if not self.atom_subset:
            return "No skills injected for this experiment condition."

        sections: List[str] = []
        task_type = retrieved_memories.get("task_type", "unknown")
        mode = retrieved_memories.get("retrieval_mode", "template")

        # General skills
        general_skills = retrieved_memories.get("general_skills", [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                text = self._assemble_skill_text(skill)
                if text.strip():
                    lines.append(text)
                    lines.append("")  # blank line between skills
            sections.append("\n".join(lines).rstrip())

        # Task-specific skills
        task_skills = retrieved_memories.get("task_specific_skills", [])
        if task_skills:
            if mode == "embedding":
                section_title = "### Task-Relevant Skills"
            else:
                task_name = task_type.replace("_", " ").title()
                section_title = f"### {task_name} Skills"
            lines = [section_title]
            for skill in task_skills:
                text = self._assemble_skill_text(skill)
                if text.strip():
                    lines.append(text)
                    lines.append("")
            sections.append("\n".join(lines).rstrip())

        # Common mistakes — format unchanged (not an ablation target)
        mistakes = retrieved_memories.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get("description", "")
                fix = mistake.get("how_to_avoid", "")
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No relevant skills found for this task."

    # ------------------------------------------------------------------ #
    # Override: _skill_to_text for embedding mode                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _skill_to_text(skill: Dict[str, Any]) -> str:
        """
        Concatenate skill fields for embedding similarity matching.

        Overrides parent to handle 6-atom format. For embedding retrieval we
        always use all fields (regardless of ablation subset) so that ranking
        is consistent across ablation conditions.
        """
        parts: List[str] = []
        for field in ("title", "strategy", "trigger"):
            val = skill.get(field, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        # Also include principle for backward compat with old-format JSONs
        principle = skill.get("principle", "")
        if isinstance(principle, str) and principle.strip():
            parts.append(principle.strip())
        when = skill.get("when_to_apply", "")
        if isinstance(when, str) and when.strip():
            parts.append(when.strip())
        return ". ".join(parts)
