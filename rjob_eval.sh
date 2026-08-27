#!/bin/bash
set -e

BASE="/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp"
PYBIN="/mnt/shared-storage-user/medeval-share/fanjiacheng/miniconda3/envs/omics/bin/python3"
NAME="cmr-irene-v7-eval-test"

CKPT="${CKPT:-outputs/cmr_irene_v7/checkpoints/best_model.pth}"
SPLIT="${SPLIT:-test}"

rjob submit \
  --priority=9 \
  --name="${NAME}" \
  --gpu=1 --cpu=16 --memory=120000 \
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
    ${PYBIN} -m multi_fusion.cmr_irene_v7.eval \
      --ckpt ${CKPT} \
      --split ${SPLIT} \
      --batch_size 64 \
      --workers 0
  "
