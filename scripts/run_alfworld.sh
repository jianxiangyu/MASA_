set -x

MODEL_SIZE="4b"   
SKILL_JSON="./memory_data/alfworld/4b_masa_skills.json"  

case "$MODEL_SIZE" in
    4b)
        MODEL_PATH="./model/Qwen3-4B"
        TP_SIZE=1; GPU_MEM=0.5; MICRO_BATCH=4; LOG_PROB_BATCH=8; REF_BATCH=4; ENFORCE_EAGER=False ;;
    8b)
        MODEL_PATH="./model/Qwen3-8B"
        TP_SIZE=1; GPU_MEM=0.5; MICRO_BATCH=4; LOG_PROB_BATCH=8; REF_BATCH=4; ENFORCE_EAGER=False ;;
    14b)
        MODEL_PATH="./model/Qwen3-14B"
        TP_SIZE=4; GPU_MEM=0.6; MICRO_BATCH=4; LOG_PROB_BATCH=8; REF_BATCH=4; ENFORCE_EAGER=False ;;
    32b)
        MODEL_PATH="./model/Qwen3-32B"
        TP_SIZE=8; GPU_MEM=0.4; MICRO_BATCH=2; LOG_PROB_BATCH=2; REF_BATCH=2; ENFORCE_EAGER=True ;;
esac

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export ALFWORLD_DATA=./data/alfworld

EMBEDDING_MODEL="./model/Qwen3-Embedding-0.6B"
val_data_size=140

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/verl-agent/text/train.parquet \
    data.val_files=./data/verl-agent/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
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
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus=0.1 \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=$SKILL_JSON \
    +env.skills_only_memory.retrieval_mode=embedding \
    +env.skills_only_memory.embedding_model_path=$EMBEDDING_MODEL \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.task_specific_top_k=5 \
    +env.skills_only_memory.enable_dynamic_update=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='alfworld_masa_skill_eval' \
    trainer.experiment_name="alfworld_qwen3_${MODEL_SIZE}_masa" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    trainer.log_val_generations=$val_data_size
