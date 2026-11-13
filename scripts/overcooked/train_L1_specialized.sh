#!/bin/bash
# Train SPECIALIZED L1 agents with different recipe subsets
# 
# 4 Types of L1 agents (based on which L0 opponents they train against):
#   1. LO_PB: Trained against L0 with LettuceOnion + PotatoBroccoli only (pool_seed=3, size=2)
#   2. LO_TC: Trained against L0 with LettuceOnion + TomatoCarrot only (pool_seed=1, size=2)
#   3. PB_TC: Trained against L0 with TomatoCarrot + PotatoBroccoli only (pool_seed=2, size=2)
#   4. ALL:   Trained against L0 with all 3 recipes (pool_seed=1, size=3)
#
# Pool seed mappings (verified with actual ingredient list):
#   pool_seed=1, size=2: ['LO', 'TC'] -> LO+TC (no PB)
#   pool_seed=2, size=2: ['TC', 'PB'] -> TC+PB (no LO)  
#   pool_seed=3, size=2: ['PB', 'LO'] -> PB+LO (no TC)
#
# 5 training seeds per type = 20 agents total
# Distributed across GPUs 0-6 (NOT GPU 7 which is running L2)
#
# Deterministic L0 opponents (p=0.0)
# RIGHT side agents (player-id=1) for use with L2 LEFT

# Activate conda environment
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pace}"

echo "=========================================================================="
echo "Training SPECIALIZED L1 RIGHT agents with recipe subsets"
echo "=========================================================================="
echo ""
echo "Agent Types (determined by L0 opponent pool):"
echo "  1. LO_PB: L0 opponents use LettuceOnion + PotatoBroccoli (no TomatoCarrot)"
echo "  2. LO_TC: L0 opponents use LettuceOnion + TomatoCarrot (no PotatoBroccoli)"
echo "  3. PB_TC: L0 opponents use PotatoBroccoli + TomatoCarrot (no LettuceOnion)"
echo "  4. ALL:   L0 opponents use all 3 recipes"
echo ""
echo "Training seeds per type: 5 (total 20 agents)"
echo "GPUs: 0-6 (excluding GPU 7 which is running L2 training)"
echo "Deterministic L0 opponents (p=0.0)"
echo "=========================================================================="

# Create directory structure
mkdir -p logs/Overcooked/L1_specialized

# Function to launch training
# Arguments: train_seed, player_id, agent_type, pool_size, pool_seed, gpu_id
launch_training() {
    local train_seed=$1
    local player_id=$2
    local agent_type=$3
    local pool_size=$4
    local pool_seed=$5
    local gpu_id=$6
    
    local exp_dir="L1_specialized/${agent_type}/seed${train_seed}"
    local log_file="logs/Overcooked/L1_specialized/${agent_type}_seed${train_seed}.log"
    
    echo "Launching: ${agent_type}_seed${train_seed} (GPU=${gpu_id}, pool_size=${pool_size}, pool_seed=${pool_seed})"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python train_.py --env-name Overcooked \
      --env_config './environment/overcooked/config/fc_small_test.yaml' \
      --player-id ${player_id} --recipe-type cross \
      --algo ppo --num-steps 1000 --num-processes 12 --use-gae --eps 1e-8 --use-proper-time-limit \
      --lr 1e-3 --entropy-coef 3e-2 --value-loss-coef 0.5 --num-epochs 4 --num-mini-batch 18 --gamma 0.99 --gae-lambda 0.95 \
      --max-grad-norm 15.0 \
      --num-env-steps 30000000 \
      --save-interval 5000000 --log-interval 10000 --eval-interval 100000 --eval-episodes 5 \
      --exp-name "${exp_dir}" \
      --latent-training --e2e-obj \
      --deterministic-latent \
      --use-latent-critic \
      --pre-hidden-dims 128 128 --encoder-base mlp --agg-func mean --post-hidden-dims 128 \
      --latent-dim 128 --kl-coef 0 \
      --hidden-dims 128 128 --act-func relu \
      --train-pool-size ${pool_size} --eval-pool-size ${pool_size} \
      --rule-based-opponents ${pool_size} \
      --pool-seed ${pool_seed} \
      --history-middle-sampling \
      --merge-encoder-computation \
      --history-full-size 5 \
      --history-size 5 \
      --history-use-episodes \
      --include-current-episode \
      --use-meta-episode \
      --self-obs-mode \
      --clear-history-on-full \
      --separate-history \
      --seed ${train_seed} \
      --p 0.0 \
      ${WANDB_USER:+--wandb-user-name ${WANDB_USER}} > ${log_file} 2>&1 &
    
    echo "$! GPU${gpu_id} ${agent_type}_seed${train_seed}" >> /tmp/l1_specialized_pids.txt
}

# Clear previous PID file
rm -f /tmp/l1_specialized_pids.txt

# GPU assignment (0-6, no GPU 7)
# 20 agents across 7 GPUs = ~3 per GPU
gpus=(0 1 2 3 4 5 6)
num_gpus=7
gpu_idx=0

echo ""
echo "======================================================================"
echo "Launching 20 L1 specialized agents across GPUs 0-6"
echo "======================================================================"

# Type 1: ALL (5 seeds) - pool_size=3, pool_seed=1 (all 3 recipes)
echo ""
echo "--- Type: ALL (L0 uses all 3 recipes) ---"
for train_seed in 1 2 3 4 5; do
    gpu=${gpus[$((gpu_idx % num_gpus))]}
    launch_training ${train_seed} 1 "ALL" 3 1 ${gpu}
    gpu_idx=$((gpu_idx + 1))
    sleep 2
done

# Type 2: LO_PB (5 seeds) - pool_size=2, pool_seed=3 
# pool_seed=3: ['PB', 'LO'] -> PB+LO (no TC)
echo ""
echo "--- Type: LO_PB (L0 uses LettuceOnion + PotatoBroccoli, no TomatoCarrot) ---"
for train_seed in 1 2 3 4 5; do
    gpu=${gpus[$((gpu_idx % num_gpus))]}
    launch_training ${train_seed} 1 "LO_PB" 2 3 ${gpu}
    gpu_idx=$((gpu_idx + 1))
    sleep 2
done

# Type 3: LO_TC (5 seeds) - pool_size=2, pool_seed=1
# pool_seed=1: ['LO', 'TC'] -> LO+TC (no PB)
echo ""
echo "--- Type: LO_TC (L0 uses LettuceOnion + TomatoCarrot, no PotatoBroccoli) ---"
for train_seed in 1 2 3 4 5; do
    gpu=${gpus[$((gpu_idx % num_gpus))]}
    launch_training ${train_seed} 1 "LO_TC" 2 1 ${gpu}
    gpu_idx=$((gpu_idx + 1))
    sleep 2
done

# Type 4: PB_TC (5 seeds) - pool_size=2, pool_seed=2
# pool_seed=2: ['TC', 'PB'] -> TC+PB (no LO)
echo ""
echo "--- Type: PB_TC (L0 uses PotatoBroccoli + TomatoCarrot, no LettuceOnion) ---"
for train_seed in 1 2 3 4 5; do
    gpu=${gpus[$((gpu_idx % num_gpus))]}
    launch_training ${train_seed} 1 "PB_TC" 2 2 ${gpu}
    gpu_idx=$((gpu_idx + 1))
    sleep 2
done

echo ""
echo "======================================================================"
echo "All 20 L1 SPECIALIZED training jobs launched!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  - ALL:   5 seeds, L0 opponents with all 3 recipes"
echo "  - LO_PB: 5 seeds, L0 opponents with LettuceOnion + PotatoBroccoli"
echo "  - LO_TC: 5 seeds, L0 opponents with LettuceOnion + TomatoCarrot"
echo "  - PB_TC: 5 seeds, L0 opponents with PotatoBroccoli + TomatoCarrot"
echo ""
echo "GPU Distribution:"
echo "  GPUs 0-6 (GPU 7 excluded - running L2 training)"
echo "  ~3 agents per GPU"
echo ""
echo "Save location: logs/Overcooked/L1_specialized/{type}/seed{1-5}/"
echo ""
echo "Directory structure:"
echo "  logs/Overcooked/L1_specialized/"
echo "    ├── ALL/"
echo "    │   ├── seed1/, seed2/, ..., seed5/"
echo "    ├── LO_PB/"
echo "    │   ├── seed1/, seed2/, ..., seed5/"
echo "    ├── LO_TC/"
echo "    │   ├── seed1/, seed2/, ..., seed5/"
echo "    └── PB_TC/"
echo "        ├── seed1/, seed2/, ..., seed5/"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/Overcooked/L1_specialized/*.log"
echo "  watch -n 1 'nvidia-smi'"
echo ""
echo "PIDs stored in: /tmp/l1_specialized_pids.txt"
cat /tmp/l1_specialized_pids.txt
echo ""
echo "======================================================================"
