#!/bin/bash
set -e

BASE="/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp"
PYBIN="/mnt/shared-storage-user/medeval-share/fanjiacheng/miniconda3/envs/omics/bin/python3"
NAME="cmr-irene-v7-end2end"

OUT_DIR="${OUT_DIR:-outputs/cmr_irene_v7_end2end_sa6t50_la1_single_stage_bb1e5_md0p1_noaocrop}"
EPOCHS="${EPOCHS:-30}"
FROZEN_EPOCHS=0
BATCH="${BATCH:-8}"
LR_HEAD="${LR_HEAD:-1e-4}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
BACKBONE_LR_ZERO_EPOCHS=0
MOD_DROP_P="${MOD_DROP_P:-0.1}"
LAMBDA_LIA="${LAMBDA_LIA:-0.1}"
LIA_TEMPERATURE="${LIA_TEMPERATURE:-0.1}"
DISABLE_LIA="${DISABLE_LIA:-0}"
SA_LA_RANDOM_CROP="${SA_LA_RANDOM_CROP:-1}"
AO_RANDOM_CROP="${AO_RANDOM_CROP:-0}"
SA_TARGET_Z="${SA_TARGET_Z:-6}"
SA_TARGET_T="${SA_TARGET_T:-50}"
LA_TARGET_Z="${LA_TARGET_Z:-1}"
LA_TARGET_T="${LA_TARGET_T:-50}"
AMP="${AMP:-auto}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
RESUME="${RESUME:-}"
GPUS="${GPUS:-4}"
CPU="${CPU:-32}"
MEMORY="${MEMORY:-240000}"
WORKERS="${WORKERS:-4}"

if [[ "${SA_LA_RANDOM_CROP}" == "1" || "${SA_LA_RANDOM_CROP}" == "true" || "${SA_LA_RANDOM_CROP}" == "TRUE" ]]; then
  SA_LA_RANDOM_CROP_ARG="--sa_la_random_crop"
else
  SA_LA_RANDOM_CROP_ARG="--no_sa_la_random_crop"
fi

if [[ "${AO_RANDOM_CROP}" == "1" || "${AO_RANDOM_CROP}" == "true" || "${AO_RANDOM_CROP}" == "TRUE" ]]; then
  AO_RANDOM_CROP_ARG="--ao_random_crop"
else
  AO_RANDOM_CROP_ARG="--no_ao_random_crop"
fi

if [[ "${GPUS}" -gt 1 ]]; then
  TRAIN_LAUNCH="${PYBIN} -m torch.distributed.run --standalone --nproc_per_node=${GPUS}"
else
  TRAIN_LAUNCH="${PYBIN}"
fi

RESUME_ARG=""
if [[ -n "${RESUME}" ]]; then
  RESUME_ARG="--resume ${RESUME}"
fi

LIA_ARG=""
if [[ "${DISABLE_LIA}" == "1" || "${DISABLE_LIA}" == "true" || "${DISABLE_LIA}" == "TRUE" ]]; then
  LIA_ARG="--disable_lia"
fi

rjob submit \
  --priority=9 \
  --name="${NAME}" \
  --gpu=${GPUS} --cpu=${CPU} --memory=${MEMORY} \
  --charged-group=evalmed_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs2/gpfs-aging:/mnt/shared-storage-gpfs2/gpfs-aging \
  --mount=gpfs://gpfs1/medeval-share:/mnt/shared-storage-user/medeval-share \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --custom-resources brainpp.cn/fuse=1 \
  --image="registry.h.pjlab.org.cn/ailab-medeval-medeval_gpu/omicgpu:jcfan-v-cu128torhc27" \
  --host-network=false \
  -e DISTRIBUTED_JOB=true \
  -e PYTHONUNBUFFERED=1 \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -- bash -c "
    mkdir -p /tmp/matplotlib && \
    cd ${BASE} && \
    ${TRAIN_LAUNCH} -m multi_fusion.cmr_irene_v7.late_gated.train_end2end \
      --out_dir ${OUT_DIR} \
      --epochs ${EPOCHS} \
      --frozen_epochs ${FROZEN_EPOCHS} \
      --batch_size ${BATCH} \
      --lr_head ${LR_HEAD} \
      --lr_backbone ${LR_BACKBONE} \
      --backbone_lr_zero_epochs ${BACKBONE_LR_ZERO_EPOCHS} \
      --modality_dropout_p ${MOD_DROP_P} \
      --lambda_lia ${LAMBDA_LIA} \
      --lia_temperature ${LIA_TEMPERATURE} \
      ${LIA_ARG} \
      ${SA_LA_RANDOM_CROP_ARG} \
      ${AO_RANDOM_CROP_ARG} \
      --sa_target_z ${SA_TARGET_Z} \
      --sa_target_t ${SA_TARGET_T} \
      --la_target_z ${LA_TARGET_Z} \
      --la_target_t ${LA_TARGET_T} \
      --amp ${AMP} \
      --early_stop_patience ${EARLY_STOP_PATIENCE} \
      --early_stop_min_delta ${EARLY_STOP_MIN_DELTA} \
      ${RESUME_ARG} \
      --workers ${WORKERS}
  "
