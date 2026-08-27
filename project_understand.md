# Project Understanding: cmr_irene_v7

这份文档用于在丢失对话上下文后快速恢复项目理解。当前目录是
`/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/multi_fusion/cmr_irene_v7`。
训练和评估应从包根目录 `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp` 以
`python -m multi_fusion.cmr_irene_v7...` 方式运行。

本次 review 对比对象是
`/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/multi_fusion/cmr_irene_v4`，
第一轮 review 按要求暂时不包括 v4 下的 `./late_gated`。随后 v7 已新增
`late_gated` 子目录，用于端到端 raw CMR 图像训练。

## 项目目标

- 当前项目关注四模态 CMR token 年龄回归：使用 IRENE 风格 Transformer 融合
  SA、LA 4ch、T1、Aortic 四类预提取 token，对 UKB 受试者的 `age_at_scan`
  做回归预测。
- 这是基于 v4 token-only 主流程继续改造的版本，不是原始 IRENE 的图像/临床
  8 类疾病分类任务。
- v7 的核心新增目标是在年龄回归主损失之外，引入 LIA
  (local inter-modal alignment / local alignment) 辅助损失，让同一受试者内不同
  存在模态的 token embedding 在局部 token 层面对齐。

## v7 相比 v4 的实际差异

代码级差异集中在以下文件：

- `models/local_alignment.py`: v7 新增文件，定义 `LocalAlignmentLoss`。
- `models/configs.py`: 新增 LIA 配置项：
  - `use_lia = True`
  - `lia_temperature = 0.1`
  - `lambda_lia = 0.1`
  - `lia_exclude_cls = True`
- `models/modeling_irene.py`: `Transformer.forward()` 增加
  `return_embedded` 参数；`CMRIrene` 在训练时可计算
  `loss = loss_reg + lambda_lia * loss_lia`，并返回 `loss_dict`。
- `train.py`: 包名和默认输出目录切到 v7；新增 CLI 参数
  `--lambda_lia`、`--lia_temperature`、`--disable_lia`；训练/验证日志额外记录
  `loss_reg`、`loss_lia`、`lia_valid_subjects`。
- `eval.py`: 包名和默认 checkpoint 切到 v7；worker 默认从 `0` 改为 `4`；
  从训练目录的 `logs/startup_metrics.json` 读取 LIA 配置以构建模型。
- `rjob_train.sh`: 作业名、输出目录和模块路径切到 v7；新增环境变量
  `LAMBDA_LIA`、`LIA_TEMPERATURE`。
- `rjob_eval.sh`: 作业名、默认 checkpoint 和模块路径切到 v7。

除上述内容外，`data/cmr_dataset.py`、`models/embed.py`、`models/encoder.py`、
`models/attention.py`、`models/block.py`、`models/mlp.py` 与 v4 主流程保持一致。

## 当前有效入口

- `train.py`: 当前训练入口。构建 train/val `CMRTokenDataset`，基于训练集年龄
  mean/std 做 z-score，训练 `CMRIrene`，用反归一化年龄计算
  MAE/RMSE/R2/Pearson r。
- `eval.py`: 当前评估入口。加载指定 checkpoint，读取对应训练目录下的
  `startup_metrics.json`，在指定 split 上评估 `CMRIrene` 并保存指标和预测 CSV。
- `rjob_train.sh`: 集群训练提交脚本，默认运行
  `multi_fusion.cmr_irene_v7.train`，输出到 `outputs/cmr_irene_v7`。
- `rjob_eval.sh`: 集群评估提交脚本，默认评估
  `outputs/cmr_irene_v7/checkpoints/best_model.pth`。
- `late_gated/train_end2end.py`: v7 端到端 IRENE 正式训练入口，模块路径是
  `multi_fusion.cmr_irene_v7.late_gated.train_end2end`。
- `late_gated/eval_end2end.py`: v7 端到端 IRENE 评估入口，模块路径是
  `multi_fusion.cmr_irene_v7.late_gated.eval_end2end`。
- `late_gated/rjob_train_end2end.sh`: v7 端到端集群训练提交脚本。
- `late_gated/rjob_eval_end2end.sh`: v7 端到端集群评估提交脚本。

## 非当前流程

- `README.md`、`irene.py`、`run.sh` 来自原始 IRENE 图像/临床分类代码，依赖 PNG
  图像、pkl 临床特征、Apex、AUROC 和 `IRENE` 分类模型。
- 当前 `models/modeling_irene.py` 定义的是 `CMRIrene`，没有定义原始 `IRENE`
  类；因此不要把 `irene.py` 或 `run.sh` 当作当前训练/验证/评估路径。

## 文件地图

- `data/cmr_dataset.py`: 当前数据集实现。负责读取 split、年龄标签、四模态 token
  文件，处理缺失模态和训练时 modality dropout，并提供 `collate_fn`。
- `models/configs.py`: 定义 `CONFIGS["CMR_IRENE"]`，包括 hidden size、token dim、
  模态数、Transformer 层数、head 数、dropout，以及 v7 新增的 LIA 配置。
- `models/embed.py`: 每个模态独立 `Linear(1024, 768)`，加独立 CLS token 和
  position embedding，把每路 `[B, 256, 1024]` 变成 `[B, 257, 768]`。
- `models/attention.py`: 实现两种 attention。前期 multi-modal block 对每个目标
  模态做 self attention 和来自其他存在模态的 cross attention；后期 concat 后做
  共享 self attention。
- `models/block.py`: Transformer block。multi-modal 阶段每模态独立
  LayerNorm/FFN；concat 阶段共享 LayerNorm/FFN。
- `models/encoder.py`: 前 `mm_layers` 层保持四路模态分开，后续层把所有模态 token
  concat，并生成 token-level mask。
- `models/local_alignment.py`: v7 新增的 LIA 辅助损失。对每个受试者、每个存在模态，
  用该模态 token 查询其他存在模态拼接后的 token context，再做双向 token-level
  cross entropy。
- `models/modeling_irene.py`: `CMRIrene` 主模型。Transformer 编码后按 mask 做 mean
  pooling，再用 MLP regression head 输出归一化年龄；训练时返回总 loss、预测和
  loss 分解。
- `late_gated/dataset.py`: 端到端 raw 四模态数据集，复用 v4 的 visit1-only 四模态
  交集、raw NPZ 读取、SA/LA 3D preprocessing、T1/AO DINO preprocessing、随机 crop
  和 flip 逻辑。
- `late_gated/model_end2end.py`: 端到端主模型。四个 raw backbone 返回 token
  sequence，经 `TokenReducer` 规整为 `[B, 256, 1024]` 后调用 v7 `CMRIrene`。
- `late_gated/train_end2end.py`: 端到端训练脚本，保留 v4 的 DDP/AMP/resume/early
  stop/多参数组/deep supervision 训练基础设施，并把 v7 LIA loss 接入主损失；当前
  项目策略固定为 single-stage soft fine-tuning。历史 true-frozen / LR-zero 参数仅
  用于兼容旧记录或手动复现实验，不再作为当前训练方案。
- `late_gated/eval_end2end.py`: 端到端 checkpoint 评估脚本，读取
  `startup_metrics.json` 中的 target mean/std、输入 shape、deep supervision 和 LIA
  配置。

## 数据契约

- split 文件: `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/data/metadata/unified_split.json`
- 年龄标签: `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/UKB_processed/age_at_scan_wide.csv`
- 目标列: `age_at_scan`
- split 中 subject 可以是字符串，也可以是包含 `subject_id` 的 dict；dataset 内统一
  转成字符串 sid。
- 标签 CSV 读取 `eid` 和目标列；缺失标签或无法转成 float 的样本会跳过。
- 模态顺序固定为 `sa`, `la`, `t1`, `aortic`；其中内部 key `la` 对应磁盘目录
  `token_embeddings/la_4ch`。
- token 根目录：
  - `sa`: `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/token_embeddings/sa`
  - `la`: `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/token_embeddings/la_4ch`
  - `t1`: `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/token_embeddings/t1`
  - `aortic`: `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/token_embeddings/aortic`
- 每个可用 `.npy` token 文件形状应为 `[256, 1024]`，预加载时保留 float16，取样时
  转成 float32。
- 一个样本至少需要一个可用模态；缺失或被 dropout 的模态用全零 `[256, 1024]`
  token 和 `mask=False` 表示。
- `collate_fn` 输出：
  - `tokens`: list of 4 tensors，每个 tensor 是 `[B, 256, 1024]`
  - `mask`: `[B, 4]` bool
  - `target`: `[B]` float32
  - `eid`: list[str]

## 模型结构

- config: hidden size `768`，token dim `1024`，模态数 `4`，multi-modal 层数 `2`，
  Transformer 总层数 `12`，heads `12`，MLP dim `3072`，dropout `0.1`。
- embedding: 每个模态独立投影到 hidden size，加独立 CLS 和 position embedding，
  输出 4 路 `[B, 257, 768]`。
- fusion:
  - 前 2 层：四路模态保持分离，做 cross-modal attention。目标模态有自己的
    self attention，并平均融合其他存在模态的 cross context。
  - 后 10 层：把四路 `[B, 257, 768]` concat 成 `[B, 4*257, 768]`，用共享 self
    attention 继续编码。
- mask:
  - multi-modal 阶段用 `[B, 4]` modality mask 屏蔽缺失目标模态和缺失 source
    模态贡献。
  - concat 阶段把 modality mask 扩展成 token mask，缺失模态的 257 个 token 都被
    屏蔽。
  - pooling 阶段按 token mask 做 masked mean pooling。
- head: `Linear(768,1024) -> LayerNorm -> GELU -> Dropout(0.2) -> Linear(1024,256)
  -> GELU -> Dropout(0.2) -> Linear(256,1)`。
- 主任务 loss: `HuberLoss`，目标是 z-score normalized age。

## LIA 辅助损失

- LIA 只在 `target is not None`、`use_lia=True` 且 `lambda_lia > 0` 时计算。
  因此 `train.py` 的 train 和 val epoch 都会计算 LIA；`eval.py` 推理时
  `target=None`，不会计算 LIA。
- LIA 使用 `Embeddings` 输出的 per-modality embedded tokens，而不是最终
  Transformer encoder 输出。默认 `lia_exclude_cls=True`，所以每个模态只用 256 个
  patch/token，不包含 CLS。
- 对每个 batch 内受试者：
  - 若存在模态数少于 2，则该受试者不参与 LIA。
  - 对每个存在模态 `m`，以 `m` 的 token 为 query，把其他存在模态 token 拼接成
    context。
  - query/context 先做 L2 normalize。
  - 用 query 对 context 做 attention 得到 attended context。
  - 再计算 query 与 attended context 的 token similarity，并做双向 cross entropy。
  - 一个受试者内对所有存在模态求平均，batch 内对有效受试者求平均。
- `CMRIrene.forward()` 训练返回：
  - `loss = loss_reg + lambda_lia * loss_lia`
  - `pred`: 归一化年龄预测
  - `loss_dict`: detached 的 `loss_reg`、`loss_lia`、`lia_valid_subjects`
- `train.py` 中 `loss_lia` 日志按有效受试者数加权平均；`loss_reg` 按 batch 样本数
  加权平均。

## 训练和评估逻辑

- 训练 split 使用 `splits=["train"]`，验证 split 使用 `splits=["val"]`；评估默认按
  参数选择 split，常用是 `test`。
- 训练集 target mean/std 来自 `ds_train.records`，模型学习归一化年龄；指标计算前把
  预测反归一化回真实年龄。
- 训练默认 modality dropout 概率是 `0.1`，对四个模态独立采样；如果会 drop 掉所有
  可用模态，则随机恢复一个原本可用模态。
- val/eval 使用 `modality_dropout_p=0.0`。
- 训练默认超参数：AdamW、lr `1e-4`、weight decay `0.01`、batch size `64`、
  epochs `30`、warmup epochs `2`、linear warmup 后 cosine decay、grad clip `1.0`。
- CUDA 下使用 `torch.amp.autocast` 和 `GradScaler`；CPU 下仍可运行但预加载和模型计算
  都比较重。
- v7 LIA 默认开启，`lambda_lia=0.1`，`lia_temperature=0.1`。可通过
  `--disable_lia` 关闭，或通过 `--lambda_lia`、`--lia_temperature` 调整。
- `startup_metrics.json` 会保存 `target_mean`、`target_std`、`modality_dropout_p`、
  `use_lia`、`lambda_lia`、`lia_temperature`；`eval.py` 依赖该文件恢复归一化和模型
  配置。

## End-to-end 训练逻辑

- v7 的 `late_gated` 从 v4 `late_gated` 复制并裁剪，只保留正式 end-to-end IRENE
  文件：`dataset.py`、`model_end2end.py`、`train_end2end.py`、`eval_end2end.py`、
  `rjob_train_end2end.sh`、`rjob_eval_end2end.sh` 和 `__init__.py`。
- 没有复制 v4 中保留的 simple late-gated baseline 文件
  `model.py`、`train.py`、`eval.py`、`rjob_train.sh`、`rjob_eval.sh`，避免把旧 gate
  baseline 当成 v7 当前入口。
- 启动目录: `/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp`。
- 推荐提交脚本:
  `bash multi_fusion/cmr_irene_v7/late_gated/rjob_train_end2end.sh`。
- 2026-07-16 已用上述脚本提交一次默认配置训练，rjob 名称:
  `cmr-irene-v7-end2end-9140590`。这是迁移 backbone LR-zero warm-up 之前的旧默认配置。
- 2026-07-17 已提交当前 backbone LR-zero warm-up 配置训练，rjob 名称:
  `cmr-irene-v7-end2end-52038375`；默认输出目录为
  `outputs/cmr_irene_v7_end2end_sa6t50_lr0e2_bb1e6`。该 job 已于 2026-07-18 停止。
- 2026-07-18 已提交新的 LA Z=6 / backbone LR-zero 1 epoch 配置训练，rjob 名称:
  `cmr-irene-v7-end2end-92709070`；默认输出目录为
  `outputs/cmr_irene_v7_end2end_sa6t50_la6_lr0e1_bb3e6`。
- 2026-07-18 历史 true-frozen 7 epoch 配置记录：当时 `rjob_train_end2end.sh` 默认
  `EPOCHS=30`、`FROZEN_EPOCHS=7`、`BACKBONE_LR_ZERO_EPOCHS=0`、`LR_HEAD=1e-4`、
  `LR_BACKBONE=1e-5`，默认输出目录为
  `outputs/cmr_irenev7_end2end_sa6t50_la6_truefrozen7_bb1e5`。
- 训练模块: `multi_fusion.cmr_irene_v7.late_gated.train_end2end`。
- 当前 rjob 默认输出目录:
  `outputs/cmr_irene_v7_end2end_sa6t50_la1_single_stage_bb1e5_md0p1_noaocrop`。
- 评估模块: `multi_fusion.cmr_irene_v7.late_gated.eval_end2end`。
- 默认评估 checkpoint:
  `outputs/cmr_irene_v7_end2end_sa6t50_la1_single_stage_bb1e5_md0p1_noaocrop/checkpoints/best_model.pth`。
  如评估其他历史产物，需要用 `CKPT` 覆盖到对应 checkpoint。
- 单模态 checkpoint:
  - SA: `outputs/clean/sax/age_at_scan/checkpoints/best_model.pth`
  - LA: `outputs/clean/lax4ch/age_at_scan/checkpoints/best_model.pth`
  - T1: `outputs/clean_dinov2/shmolli/checkpoints/best_model.pth`
  - AO: `outputs/clean_dinov2/aortic/checkpoints/best_model.pth`
- 数据 split: `visit1only_sax_split.json`、`visit1only_lax4ch_split.json`、
  `visit1only_shmolli_split.json`、`visit1only_ao_split.json` 的四模态交集。
- 默认输入 shape:
  - SA: `[1, 6, 50, 128, 128]`
  - LA: `[1, 1, 50, 128, 128]`
  - T1: `[3, 224, 224]`
  - AO: `[3, 224, 224]`
- Random crop 默认开启：
  - SA/LA 训练时对整个 `[Z,T,H,W]` volume 做同一个 spatial random crop 到
    `128x128`；val/test 使用 center crop。Z/T 不随机裁剪，按 `target_z` center
    crop/pad，按 `target_t` 线性采样/pad。
  - AO 默认先 resize 到 `256x256`，训练 random crop 到 `224x224`；val/test 使用
    center crop。T1 不走 random crop，固定取 best slice 的 `[0,3,6]` frames 后
    resize 到 `224x224`。
- 端到端模型结构:
  1. 四个 raw backbone 分别产生 token sequence 和 aux features。
  2. `TokenReducer` 把不同 backbone token 统一到 `[B, 256, 1024]`。
  3. 训练时生成 `[B,4]` modality dropout mask，并把 dropped token 置零。
  4. `EndToEndMultiModalAgeRegressor.forward(inputs, target=tgt_n)` 把 tokens、mask 和
     normalized target 传给 v7 `CMRIrene`。
  5. v7 `CMRIrene` 返回 `irene_loss = loss_reg + lambda_lia * loss_lia`、预测和
     `loss_dict`。
  6. 总训练 loss 为 `irene_loss + lambda_aux * aux_loss`。
- 默认端到端训练参数: epochs `30`，per-GPU batch size `8`，GPUs `4`，
  global batch size `32`，workers `4`。
- 优化器: AdamW，head/token bridge/IRENE/aux heads lr `1e-4`；direct
  `train_end2end.py` 默认 `--frozen_epochs=0`、backbone lr `1e-5` 且
  `--backbone_lr_zero_epochs=0`。当前 rjob 默认已改为 paired-control single-stage 配置：
  `FROZEN_EPOCHS=0`、`BACKBONE_LR_ZERO_EPOCHS=0`、`LR_BACKBONE=1e-5`；
  weight decay `1e-4`，grad clip `1.0`。
- 训练策略: single-stage soft fine-tuning，这是当前及后续采用的正式策略；不再使用
  2-stage / true-frozen / backbone LR-zero warm-up 训练。`FROZEN_EPOCHS=0` 且
  `BACKBONE_LR_ZERO_EPOCHS=0`，所以 backbone 从 epoch 1 进入 optimizer，并以
  `lr_backbone=1e-5` 更新。rjob 脚本固定传入这两个 0 值，避免环境变量误把训练切回
  历史 2-stage 配置。`train_end2end.py` 中的 `--frozen_epochs` 和
  `--backbone_lr_zero_epochs` 仍保留为历史兼容开关；当前新实验不使用。
  `epoch_metrics.jsonl` 会记录 `stage`、`backbone_trainable` 和
  `optimizer_includes_backbone`，当前配置下 `stage` 应为 `soft_finetune`。
- Deep supervision 默认开启；只有传 `--no_deep_sup` 才关闭。aux loss 权重默认
  `0.05`。每个模态使用 `-4`、`-2` 两个 aux feature，共 8 个 aux head；aux loss
  用 normalized age 的 `SmoothL1Loss`，先对 aux heads 平均，再按
  `loss = irene_loss + lambda_aux * aux_loss` 加入总 loss。历史 true-frozen / LR-zero
  阶段会使用 `lambda_aux_frozen`，当前 single-stage soft fine-tuning 使用
  `lambda_aux_unfrozen`；二者默认都为 `0.05`。
- Modality dropout 默认 `0.1`；val/eval 通过 `model.eval()` 禁用随机 dropout。
- v7 LIA 默认开启，`lambda_lia=0.1`，`lia_temperature=0.1`。端到端训练同样支持
  `--disable_lia`、`--lambda_lia`、`--lia_temperature`；rjob 脚本对应环境变量是
  `DISABLE_LIA`、`LAMBDA_LIA`、`LIA_TEMPERATURE`。
- AMP 默认 `auto`：CUDA bf16 可用时用 bf16，否则用 fp16 + GradScaler；也可设
  `--amp bf16|fp16|off`。
- DDP 自动由 `WORLD_SIZE > 1` 启用。rjob 默认 `GPUS=4`，通过
  `torch.distributed.run --standalone --nproc_per_node=${GPUS}` 启动；DDP 使用
  `find_unused_parameters=True`，训练 sampler `drop_last=True`，验证使用无 padding
  的 `DistributedEvalSampler` 避免重复样本。
- Early stopping: direct `train_end2end.py` 默认 patience `8`，当前 rjob 默认
  patience `8`，min_delta `0.0`；每个 epoch 保存 `resume_state.pth`，最佳模型保存为
  `best_model.pth`。
- `startup_metrics.json` 会保存 target mean/std、SA/LA shape、AO crop、deep
  supervision、AMP/DDP、LIA 配置以及 backbone 训练策略字段：
  `training_strategy`、`frozen_epochs`、`backbones_trainable_from_epoch`、
  `backbone_optimizer_includes_from_epoch`、`backbone_lr_zero_epochs`、
  `backbone_lr_after_zero`、`backbone_weight_updates_from_epoch`；端到端 eval 依赖该文件
  恢复模型构建参数。
- 当前 v7 `rjob_train_end2end.sh` 已确认从 v4 切到 v7 module/output，并传入关键
  环境变量：`GPUS`、`BATCH`、`OUT_DIR`、`LR_HEAD`、`LR_BACKBONE`、
  `MOD_DROP_P`、`LAMBDA_LIA`、`LIA_TEMPERATURE`、`DISABLE_LIA`、
  `SA_LA_RANDOM_CROP`、`AO_RANDOM_CROP`、`SA_TARGET_Z/T`、`LA_TARGET_Z/T`、`AMP`、
  early stop、`RESUME`、`WORKERS`。`FROZEN_EPOCHS` 和
  `BACKBONE_LR_ZERO_EPOCHS` 在 rjob 中固定为 `0`，不再接受环境变量覆盖。
- Current paired-control config as of 2026-07-20: `late_gated/rjob_train_end2end.sh` and
  `train_end2end.py` defaults were updated to the requested single-stage soft fine-tuning
  config: `OUT_DIR=outputs/cmr_irene_v7_end2end_sa6t50_la1_single_stage_bb1e5_md0p1_noaocrop`,
  `EPOCHS=30`, `FROZEN_EPOCHS=0`, `BACKBONE_LR_ZERO_EPOCHS=0`, `LR_HEAD=1e-4`,
  `LR_BACKBONE=1e-5`, `MOD_DROP_P=0.1`, `AO_RANDOM_CROP=0`,
  `SA_LA_RANDOM_CROP=1`, `SA=(Z=6,T=50)`, `LA=(Z=1,T=50)`, per-GPU `BATCH=8`,
  `GPUS=4`, `AMP=auto`, `EARLY_STOP_PATIENCE=8`. v7-specific LIA remains enabled with
  `lambda_lia=0.1` and `lia_temperature=0.1`. This v7 paired-control job has not yet been
  submitted in this record.
- Data split alignment check on 2026-07-18: `multi_fusion.cmr_irene_v7.late_gated.dataset`
  and `multi_fusion.simple_late_gated_v2.dataset` instantiate exactly the same records for the
  paired-control raw end-to-end setup. The current `LA=(Z=1,T=50)` change only affects VITA volume
  preprocessing shape, not record membership or `npz_path` selection. Train/val/test records
  match exactly by subject id, target, and all four modality `npz_path` values. Counts and target
  stats are: train `n=59592`, mean `67.133348603839`, std `7.773022815300`; val `n=8507`,
  mean `67.291519924768`, std `7.745648170108`; test `n=16988`, mean `67.055104779845`,
  std `7.731264666175`.
- 当前 v7 `rjob_eval_end2end.sh` 默认 checkpoint/module 也已切到 v7；shape 和 AO crop
  默认从训练目录的 `logs/startup_metrics.json` 自动恢复，必要时可通过环境变量覆盖。

## 已知风险 / 注意事项

- LIA 实现在 Python 层逐受试者、逐模态循环，且每个 query/context 都会构建
  token similarity；相比 v4 会显著增加训练和验证耗时。
- 端到端训练里 LIA 也会生效，因为 `train_end2end.py` 会把 normalized target 传给
  `EndToEndMultiModalAgeRegressor.forward(..., target=tgt_n)`，从而触发 v7
  `CMRIrene` 的训练分支。
- `train_end2end.py` 中仍保留历史 true-frozen / LR-zero 分支；当前正式 rjob 不触发这些
  分支。新实验启动后应在 `startup_metrics.json` 中看到
  `training_strategy=single_stage_soft_finetune_irene_fusion`，并在 epoch 日志中看到
  `stage=soft_finetune`。
- LIA 只对当前 batch 中存在至少两个模态的受试者生效；如果 modality dropout 或真实
  缺失导致大量样本只剩一个模态，`lia_valid_subjects` 会下降，LIA 实际约束会变弱。
- `train.py` 里 `total_loss` 累计后没有直接用于日志；当前日志中的 `loss` 是用
  epoch 级 `loss_reg + lambda_lia * loss_lia` 重算得到，语义上更像分量平均后的总损失，
  不一定逐点等同于每个 batch 训练 loss 的简单样本均值。
- `eval.py` 要求 checkpoint 上两级目录下存在 `logs/startup_metrics.json`。如果手动移动
  `best_model.pth` 而没有移动日志文件，评估会失败。
- `CONFIGS["CMR_IRENE"]` 返回的是一个已创建的 config 对象；训练/评估脚本会直接修改
  其 LIA 字段。当前单进程脚本没有问题，但在同一 Python 进程内多次构建不同配置时要
  注意共享对象带来的状态残留。
- 当前目录含有 `__pycache__/` 这类运行残留文件；它们不是项目逻辑的一部分。
