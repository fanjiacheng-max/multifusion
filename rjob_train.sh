#!/bin/bash
set -e

BASE="/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp"
PYBIN="/mnt/shared-storage-user/medeval-share/fanjiacheng/miniconda3/envs/omics/bin/python3"
NAME="cmr_irene_v7_train"

# 可通过环境变量覆盖输出目录和超参数
OUT_DIR="${OUT_DIR:-outputs/cmr_irene_v7}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-4}"
LAMBDA_LIA="${LAMBDA_LIA:-0.1}"
LIA_TEMPERATURE="${LIA_TEMPERATURE:-0.1}"

rjob submit \
  --priority=9 \
  --name="${NAME}" \
  --gpu=1 --cpu=16 --memory=200000 \
  --charged-group=evalmed_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs2/gpfs-aging:/mnt/shared-storage-gpfs2/gpfs-aging \
  --mount=gpfs://gpfs1/fanjiacheng:/mnt/shared-storage-user/fanjiacheng \
  --mount=gpfs://gpfs1/medeval-share:/mnt/shared-storage-user/medeval-share \
  --custom-resources brainpp.cn/fuse=1 \
  --image="registry.h.pjlab.org.cn/ailab-medeval-medeval_gpu/omicgpu:jcfan-v-cu128torhc27" \
  --host-network=false \
  -e DISTRIBUTED_JOB=false \
  -e PYTHONUNBUFFERED=1 \
  -- bash -c "
    cd ${BASE} && \
    ${PYBIN} -m multi_fusion.cmr_irene_v7.train \
      --out_dir ${OUT_DIR} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH} \
      --lr ${LR} \
      --lambda_lia ${LAMBDA_LIA} \
      --lia_temperature ${LIA_TEMPERATURE} \
      --num_workers 8
  "
