#!/bin/bash
# 실험 8: LaVAN 공격 면적 스윕 x P=8/16/32
# baseline(01_baseline.sh)은 patch_ratio=0.02(2%) 고정인데, 이 비율이 커지거나 작아짐에 따라
# P=8 vs P=16/32의 강건성 순위가 뒤바뀌는 지점(crossover)이 있는지 확인한다.
# ratio=0.02(2%)는 이미 baseline에 있으므로 나머지 4개 지점만 새로 돈다.
# 시드별로 여러 번 제출: sbatch --export=SEED=42 scripts/06_lavan_areasweep.sh
#SBATCH --job-name=vit_06_lavan_sweep
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=results/job_logs/nohup_06_lavan_sweep_%j.txt

source /home/jooyumm/ViT_robust/ViT_tradeoff/scripts/common.sh

# "tag suffix:patch_ratio" 쌍. tag는 영숫자만 가능해서 permille(천분율)로 인코딩
# (0.5%=5, 1%=10, 5%=50, 10%=100 — 2%=20는 baseline과 겹쳐서 뺌)
RATIOS=("5:0.005" "10:0.01" "50:0.05" "100:0.10")

for P in 8 16 32; do
  BS=16; if [ "$P" != "8" ]; then BS=64; fi
  for pair in "${RATIOS[@]}"; do
    TAG_SUFFIX="${pair%%:*}"
    RATIO="${pair##*:}"
    python experiments/main.py \
      --attacks lavan \
      --patch_sizes "$P" \
      --num_samples 1000 \
      --batch_size "$BS" \
      --lavan_patch_ratio "$RATIO" \
      --tag "lavanarea${TAG_SUFFIX}" \
      --seed "$SEED"
  done
done