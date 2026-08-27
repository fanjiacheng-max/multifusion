#!/bin/bash
set -e

BASE="/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp"
PYBIN="/mnt/shared-storage-user/medeval-share/fanjiacheng/miniconda3/envs/omics/bin/python3"
NAME="cmr-irene-v7-end2end-eval-test"

CKPT="${CKPT:-outputs/cmr_irene_v7_end2end_sa6t50_la1_single_stage_bb1e5_md0p1_noaocrop/checkpoints/best_model.pth}"
SPLIT="${SPLIT:-test}"
AO_RANDOM_CROP="${AO_RANDOM_CROP:-auto}"
SA_TARGET_Z="${SA_TARGET_Z:-auto}"
SA_TARGET_T="${SA_TARGET_T:-auto}"
LA_TARGET_Z="${LA_TARGET_Z:-auto}"
LA_TARGET_T="${LA_TARGET_T:-auto}"

if [[ "${AO_RANDOM_CROP}" == "auto" || "${AO_RANDOM_CROP}" == "AUTO" ]]; then
  AO_RANDOM_CROP_ARG=""
elif [[ "${AO_RANDOM_CROP}" == "1" || "${AO_RANDOM_CROP}" == "true" || "${AO_RANDOM_CROP}" == "TRUE" ]]; then
  AO_RANDOM_CROP_ARG="--ao_random_crop"
else
  AO_RANDOM_CROP_ARG="--no_ao_random_crop"
fi

SHAPE_ARGS=""
if [[ "${SA_TARGET_Z}" != "auto" && "${SA_TARGET_Z}" != "AUTO" ]]; then
  SHAPE_ARGS="${SHAPE_ARGS} --sa_target_z ${SA_TARGET_Z}"
fi
if [[ "${SA_TARGET_T}" != "auto" && "${SA_TARGET_T}" != "AUTO" ]]; then
  SHAPE_ARGS="${SHAPE_ARGS} --sa_target_t ${SA_TARGET_T}"
fi
if [[ "${LA_TARGET_Z}" != "auto" && "${LA_TARGET_Z}" != "AUTO" ]]; then
  SHAPE_ARGS="${SHAPE_ARGS} --la_target_z ${LA_TARGET_Z}"
fi
if [[ "${LA_TARGET_T}" != "auto" && "${LA_TARGET_T}" != "AUTO" ]]; then
  SHAPE_ARGS="${SHAPE_ARGS} --la_target_t ${LA_TARGET_T}"
fi

rjob submit \
  --priority=9 \
  --name="${NAME}" \
  --gpu=1 --cpu=16 --memory=120000 \
  --charged-group=evalmed_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs2/gpfs-aging:/mnt/shared-storage-gpfs2/gpfs-aging \
  --mount=gpfs://gpfs1/fanjiacheng:/mnt/shared-storage-user/fanjiacheng \
  --mount=gpfs://gpfs1/medeval-share:/mnt/shared-storage-user/medeval-share \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --custom-resources brainpp.cn/fuse=1 \
  --image="registry.h.pjlab.org.cn/ailab-medeval-medeval_gpu/omicgpu:jcfan-v-cu128torhc27" \
  --host-network=false \
  -e DISTRIBUTED_JOB=false \
  -e PYTHONUNBUFFERED=1 \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -- bash -c "
    mkdir -p /tmp/matplotlib && \
    cd ${BASE} && \
    ${PYBIN} -m multi_fusion.cmr_irene_v7.late_gated.eval_end2end \
      --ckpt ${CKPT} \
      --split ${SPLIT} \
      ${AO_RANDOM_CROP_ARG} \
      ${SHAPE_ARGS} \
      --batch_size 16 \
      --workers 4
  "
