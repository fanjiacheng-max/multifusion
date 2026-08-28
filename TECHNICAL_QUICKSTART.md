# MultiFusion 技术快速上手

本文面向第一次接手本项目的开发者，补充 README 中不适合展开的工程细节、数据契约、张量形状、运行前检查和常见故障。

## 1. 先记住这几件事

1. 当前任务是四模态 CMR 年龄回归，不是原始 IRENE 的肺部疾病分类。
2. 有两条有效流程：预提取 token 训练，以及原始图像端到端训练。
3. 根目录 `irene.py` 和 `run.sh` 是原始 IRENE 历史文件，不是当前入口。
4. 代码依赖内部集群目录、外部 ViTa、DINOv2 权重、单模态 checkpoint 和医学数据，单独克隆仓库不能直接完成训练。
5. GitHub 仓库名是 `multifusion`，但 Python 包路径仍是 `multi_fusion.cmr_irene_v7`。
6. 评估不仅需要模型 checkpoint，还需要同一实验目录下的 `logs/startup_metrics.json`。
7. 当前没有 requirements、自动化测试或小样本 smoke-test 模式，首次运行前必须手动完成数据和依赖检查。

## 2. 应该选择哪条流程

| 需求 | 使用入口 | 特点 |
| --- | --- | --- |
| 快速训练融合层 | `train.py` | 使用预提取 token，不更新图像 backbone |
| 评估 token 模型 | `eval.py` | 输出真实年龄尺度的指标和预测 CSV |
| 联合微调四个 backbone | `late_gated/train_end2end.py` | 计算量和显存占用很高，支持 DDP/AMP/resume |
| 评估端到端模型 | `late_gated/eval_end2end.py` | 仍需要外部 backbone 定义和初始化权重 |
| 复现原始 IRENE 分类 | 不属于当前可用流程 | 当前 `models/modeling_irene.py` 不再定义原始 `IRENE` 类 |

## 3. 推荐的目录放置方式

代码中的绝对导入要求项目位于 `multi_fusion/cmr_irene_v7` 包路径下。即使远程仓库名已经改为 `multifusion`，集群上的目录名仍建议保持不变：

```text
/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/
├── multi_fusion/
│   ├── __init__.py
│   └── cmr_irene_v7/          # 本仓库内容
├── data/
├── UKB_processed/
├── token_embeddings/
├── outputs/
├── ViTa/
└── multimode_train/
```

如需重新克隆：

```bash
cd /mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp/multi_fusion
gh repo clone fanjiacheng-max/multifusion cmr_irene_v7
cd /mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp
```

所有 `python -m multi_fusion.cmr_irene_v7...` 命令都应从 `cmr_tmp` 包根目录运行，而不是从仓库内部直接运行。

## 4. Token-only 数据契约

### 4.1 固定数据位置

`data/cmr_dataset.py` 当前使用以下路径：

```text
BASE=/mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp

data/metadata/unified_split.json
UKB_processed/age_at_scan_wide.csv
token_embeddings/sa/<eid>.npy
token_embeddings/la_4ch/<eid>.npy
token_embeddings/t1/<eid>.npy
token_embeddings/aortic/<eid>.npy
```

代码没有命令行参数用于覆盖 token 根目录。迁移环境时需要修改 `BASE`、`TOKEN_ROOTS`，或者将这些路径改为 CLI/config 参数。

### 4.2 Split 格式

split JSON 至少包含 `train`、`val`、`test` 中需要使用的键。每项可以是字符串 subject id，也可以是：

```json
{
  "subject_id": "1234567"
}
```

subject id 最终统一转换成字符串，并用作 token 文件名和标签表中的 `eid`。

### 4.3 标签格式

CSV 至少需要：

```csv
eid,age_at_scan
1234567,68.4
```

缺少标签、空值或无法转换为 float 的样本会被跳过。代码没有对空训练集、零方差标签或异常年龄范围做显式检查。

### 4.4 Token 文件格式

每个 `.npy` 文件应满足：

```text
shape: [256, 1024]
recommended dtype: float16
filename: <eid>.npy
```

数据集当前没有 shape 校验。错误形状通常会在 `Linear(1024,768)` 或 batch stack 阶段才报错。

样本只要存在至少一个模态就会进入数据集。缺失模态返回全零 `[256,1024]` 和 `mask=False`。

### 4.5 内存行为

`train.py` 会分别把完整 train 和 val token 预加载到内存：

- 缓存中保持 float16。
- `__getitem__` 时转换为 float32。
- DataLoader 使用 persistent workers。

因此 token-only 并不等于低内存。启动前应确认节点主存足以同时容纳训练集、验证集、Python 对象和 worker 开销。

## 5. End-to-end 原始数据契约

### 5.1 Split 文件

端到端数据集读取四个独立 split：

```text
visit1only_sax_split.json
visit1only_lax4ch_split.json
visit1only_shmolli_split.json
visit1only_ao_split.json
```

每条记录必须包含：

```json
{
  "subject_id": "1234567",
  "npz_path": "/absolute/path/to/sample.npz"
}
```

最终样本集合是同一 split 下四模态 subject id 的交集，并且 subject 必须存在年龄标签。因此该流程没有真实缺失模态；缺失模态仅由训练时 modality dropout 模拟。

### 5.2 NPZ 格式

每个 NPZ 必须含有：

```python
np.load(path)["volume"]
```

代码假设 volume 轴顺序为：

```text
[H, W, Z, T]
```

关键隐含要求：

- T1 固定读取时间帧 `[0, 3, 6]`，因此 `T >= 7`。
- AO 固定读取时间帧 `[0, 33, 66]`，因此 `T >= 67`。
- AO 固定使用 `Z=0`。
- SA/LA 会对 Z 做中心裁剪或补零，对 T 做均匀采样或尾部补零。
- SA/LA 空间 crop 对整个 `[Z,T]` volume 使用同一个 crop 区域。

### 5.3 默认模型输入

```text
SA: [B, 1, 6, 50, 128, 128]
LA: [B, 1, 1, 50, 128, 128]
T1: [B, 3, 224, 224]
AO: [B, 3, 224, 224]
```

模型内部的 ViTa wrapper 会在收到六维 SA/LA 输入时移除通道维。

### 5.4 数据读取失败的当前行为

`MultiModalRawDataset` 会捕获任意异常，然后随机换一个样本重试。这意味着：

- 错误路径和坏 NPZ 不会输出原始异常。
- train 中可能产生额外重复样本。
- val/test 中也可能用其他 subject 替换失败样本，影响评估集合完整性。

正式实验前应单独扫描全部 NPZ；不要把 dataset 的 retry 机制当作数据质量控制。

## 6. 核心张量流

### 6.1 Token embedding

每个模态使用独立参数：

```text
[B,256,1024]
  → Linear(1024,768)
  → prepend modality-specific CLS
  → add modality-specific position embedding
  → [B,257,768]
```

token 数 256 在 `models/embed.py` 中硬编码。改变 token 数时必须同时修改 position embedding 长度和端到端 TokenReducer 配置。

### 6.2 前两层跨模态融合

每个目标模态拥有独立 Q/K/V/out projection：

```text
target output = 0.5 × self_context + 0.5 × mean(valid cross_contexts)
```

如果没有其他可用模态，则只使用 self context。目标模态缺失时，attention 和 FFN 残差输出最终会被 mask 为零。

### 6.3 后十层统一编码

四路 token 拼接：

```text
4 × [B,257,768] → [B,1028,768]
```

之后使用共享 self-attention。modality mask 会扩展为 token mask，将缺失模态的 257 个位置全部屏蔽。

### 6.4 Pooling 与预测

模型对全部有效 token 做 masked mean pooling，而不是只使用 CLS：

```text
[B,1028,768] → [B,768] → MLP → [B]
```

预测值是 z-score 年龄。指标计算前执行：

```text
age = pred_norm × target_std + target_mean
```

## 7. 损失函数

### 7.1 Token-only

```text
loss = Huber(pred_norm, target_norm)
     + lambda_lia × LocalAlignmentLoss
```

### 7.2 End-to-end

```text
irene_loss = loss_reg + lambda_lia × loss_lia
total_loss = irene_loss + lambda_aux × loss_aux
```

Deep supervision 默认对四模态各取 `-4`、`-2` 两个中间特征，共 8 个辅助年龄头。

### 7.3 LIA 触发条件

LIA 仅在以下条件同时满足时运行：

- `target is not None`
- `use_lia=True`
- `lambda_lia > 0`
- 当前受试者至少保留两个模态

因此 train 和 val 会计算 LIA，纯评估 `target=None` 不会计算。

LIA 当前逐受试者、逐模态执行较大的 token similarity matmul，是明显的训练耗时和显存热点。排查速度或 OOM 时，优先用 `--disable_lia` 做对照。

## 8. 配置从哪里来

### 8.1 模型默认值

基础配置位于 `models/configs.py`：

```text
hidden_size=768
token_dim=1024
n_modalities=4
mm_layers=2
num_layers=12
num_heads=12
mlp_dim=3072
modality_dropout_p=0.1
lambda_lia=0.1
lia_temperature=0.1
```

Token-only 脚本直接修改全局 `CONFIGS["CMR_IRENE"]` 对象。在同一 Python 进程中构建多套不同配置时，应先 deepcopy，避免配置残留。端到端模型已经使用 deepcopy。

### 8.2 启动配置日志

训练会创建：

```text
<out_dir>/logs/startup_metrics.json
```

其中包含：

- 训练集 target mean/std
- LIA 开关、权重和 temperature
- modality dropout
- 端到端输入 shape
- deep supervision
- AMP/DDP 和 backbone 策略

评估使用该文件重建模型。移动 checkpoint 时必须同步移动对应的 `logs` 目录。

## 9. 第一次运行前检查

### 9.1 Git/包路径

```bash
cd /mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp
python -c "import multi_fusion.cmr_irene_v7; print('package import ok')"
```

### 9.2 Token-only 依赖

```bash
test -f data/metadata/unified_split.json
test -f UKB_processed/age_at_scan_wide.csv
test -d token_embeddings/sa
test -d token_embeddings/la_4ch
test -d token_embeddings/t1
test -d token_embeddings/aortic
```

建议额外抽样检查每种模态的文件：

```python
import numpy as np

array = np.load("token_embeddings/sa/<eid>.npy")
assert array.shape == (256, 1024)
assert np.isfinite(array).all()
print(array.dtype, array.min(), array.max())
```

### 9.3 End-to-end 额外依赖

确认以下内容存在：

- `ViTa/src/models/vita_downstream.py`
- `multimode_train/train_four_modal_mult.py`
- DINOv2 `config.json` 和 `model.safetensors`
- SA/LA/T1/AO 四个单模态 `best_model.pth`
- 四个 visit1-only split 中引用的全部 NPZ

注意：即使加载完整端到端 checkpoint，评估脚本构建模型时仍会先加载四个单模态 checkpoint 和 DINO 基础权重，因此这些文件仍必须存在。

## 10. 常用命令

所有命令从以下目录启动：

```bash
cd /mnt/shared-storage-gpfs2/gpfs-aging/cmr_tmp
```

### 10.1 Token-only 训练

```bash
python -m multi_fusion.cmr_irene_v7.train \
  --out_dir outputs/cmr_irene_v7 \
  --epochs 30 \
  --batch_size 64 \
  --lr 1e-4 \
  --num_workers 8 \
  --lambda_lia 0.1 \
  --lia_temperature 0.1
```

速度/显存排查版本：

```bash
python -m multi_fusion.cmr_irene_v7.train \
  --out_dir outputs/cmr_irene_v7_debug \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --disable_lia
```

该命令仍会预加载完整 train/val，并遍历完整 epoch，不是真正的小样本 smoke test。

### 10.2 Token-only 评估

```bash
python -m multi_fusion.cmr_irene_v7.eval \
  --ckpt outputs/cmr_irene_v7/checkpoints/best_model.pth \
  --split test \
  --batch_size 64 \
  --workers 4
```

### 10.3 End-to-end 训练

推荐通过集群脚本：

```bash
bash multi_fusion/cmr_irene_v7/late_gated/rjob_train_end2end.sh
```

默认正式策略是 single-stage soft fine-tuning：

- `frozen_epochs=0`
- `backbone_lr_zero_epochs=0`
- head lr `1e-4`
- backbone lr `1e-5`
- per-GPU batch `8`
- 默认 `4` GPUs
- AMP `auto`

### 10.4 End-to-end 恢复训练

```bash
python -m multi_fusion.cmr_irene_v7.late_gated.train_end2end \
  --out_dir <same-output-dir> \
  --resume <same-output-dir>/checkpoints/resume_state.pth
```

resume 保存模型、优化器、scaler、epoch 和 early-stop 状态，但没有保存 Python/NumPy/Torch/CUDA RNG 状态，因此不是严格 bitwise reproducible resume。

### 10.5 End-to-end 评估

```bash
python -m multi_fusion.cmr_irene_v7.late_gated.eval_end2end \
  --ckpt <experiment>/checkpoints/best_model.pth \
  --split test \
  --batch_size 16 \
  --workers 4
```

输入 shape、AO crop、deep supervision 和 LIA 配置默认从 `startup_metrics.json` 恢复。

## 11. 输出与 checkpoint 语义

```text
<out_dir>/
├── checkpoints/
│   ├── best_model.pth
│   ├── last_model.pth
│   └── resume_state.pth
└── logs/
    ├── startup_metrics.json
    └── epoch_metrics.jsonl
```

- `best_model.pth`：按验证集 MAE 选择的纯 `state_dict`。
- `last_model.pth`：训练停止时的纯 `state_dict`。
- `resume_state.pth`：仅端到端流程，包含模型和优化器等完整恢复状态。
- `startup_metrics.json`：模型重建和反归一化所需配置。
- `epoch_metrics.jsonl`：逐 epoch 指标。

评估输出：

```text
eval_<split>/test_metrics.json
eval_<split>/predictions.csv
```

即使评估的 split 不是 test，指标文件名目前仍固定为 `test_metrics.json`。

## 12. 资源和性能预期

### Token-only

- 核心 `CMRIrene` 约 132.6M 参数。
- 后十层处理长度 1028 的序列，attention 显存与 `1028²` 成正比。
- 默认 batch 64 对单 GPU 要求很高。
- token 预加载可能占用百 GB 级主存。
- LIA 会显著增加训练和验证耗时。

### End-to-end

- 同时包含两个 ViTa 和两个 DINOv2 backbone。
- Modality dropout 在 backbone 计算后执行，不能节省 backbone 计算。
- 默认四卡 DDP，每卡 batch 8，global batch 32。
- 优先使用 bf16；不支持时才使用 fp16 + GradScaler。

遇到 OOM 时依次尝试：

1. 降低 per-GPU batch size。
2. 使用 bf16/AMP。
3. 临时 `--disable_lia` 定位 LIA 开销。
4. 关闭 deep supervision 定位 aux 开销。
5. 检查是否误用了异常大的 SA/LA Z/T shape。

## 13. 常见故障

| 现象 | 首先检查 |
| --- | --- |
| `No module named multi_fusion` | 是否从 `cmr_tmp` 启动，仓库是否位于 `multi_fusion/cmr_irene_v7` |
| `cannot import name IRENE` | 是否误运行了历史 `irene.py` |
| 找不到 `startup_metrics.json` | checkpoint 是否脱离原实验目录单独移动 |
| `Linear` shape mismatch | token 是否严格为 `[256,1024]` |
| DataLoader 随机报错或评估样本重复 | 检查 raw NPZ 路径和被吞掉的读取异常 |
| T1/AO index out of bounds | T1 是否 `T>=7`，AO 是否 `T>=67` |
| ViTa checkpoint missing/unexpected keys | checkpoint variant 和 SA/LA `img_shape` 是否匹配 |
| DINO 权重加载失败 | `DINO_PATH`、transformers/safetensors 版本和文件完整性 |
| 训练极慢 | LIA、1028-token attention、数据 IO、ViTa token 数 |
| resume 后结果不完全一致 | 当前未保存 RNG 状态，随机增强和 dropout 序列会变化 |
| eval 指标出现 NaN | 空 split、恒定标签、非有限预测或数据被大量跳过 |

## 14. 常见修改应该改哪里

### 修改数据路径

- Token-only：`data/cmr_dataset.py` 的 `BASE` 和 `TOKEN_ROOTS`。
- End-to-end split：`late_gated/dataset.py` 的 `BASE` 和 `SPLIT_JSONS`。
- 单模态 checkpoint：`late_gated/train_end2end.py`、`eval_end2end.py`。
- DINO 权重：`late_gated/model_end2end.py` 的 `DINO_PATH`。

长期建议将这些路径集中到 YAML/CLI，而不是继续增加硬编码。

### 修改 token 数或维度

- `models/embed.py`：token 数当前硬编码为 256。
- `models/configs.py`：`token_dim`。
- `late_gated/model_end2end.py`：TokenReducer 的输出 token 数和维度。
- 已训练 checkpoint 的 position embedding 将不再直接兼容。

### 增加或减少模态

至少需要同步修改：

- `MOD_KEYS` / `MODALITIES`
- `n_modalities`
- dataset collate 和 mask 形状
- embedding ModuleList 数量
- end-to-end `apply_modality_dropout` 中硬编码的 4
- backbone、TokenReducer 和 aux head 构造

### 修改回归目标

- 标签 CSV 和目标列。
- 数据集 `target_col`。
- 训练/评估脚本中的标签路径。
- 输出 head 当前固定为标量回归。
- 指标仍按连续变量计算。

### 修改融合策略

- Cross/self attention：`models/attention.py`
- block 残差和 FFN：`models/block.py`
- 前后层分界：`models/encoder.py` 和 `config.mm_layers`
- pooling/head/loss：`models/modeling_irene.py`
- LIA：`models/local_alignment.py`

## 15. 当前最重要的技术风险

1. Raw dataset 在 val/test 读取失败时随机替换样本，可能改变评估集合。
2. 四模态水平翻转分别随机采样，可能破坏同一 subject 的跨模态空间一致性。
3. LIA 的正样本由 query 自己检索出的 attended context 构造，存在结构性捷径风险。
4. LIA Python 循环和大矩阵乘法是显著性能热点。
5. 路径、外部模块和 checkpoint 高度依赖内部集群环境。
6. 没有统一随机种子，resume 也没有保存 RNG 状态。
7. 没有数据 shape/finite-value 预检和自动化测试。
8. Token-only 训练没有 DDP、resume、early stopping 或真正的小样本 debug 模式。
9. 配置和 checkpoint 没有完全自包含，评估依赖旁路 JSON 和外部权重。
10. README 中建议的 PyTorch 2.x 才与当前 `torch.amp`、`weights_only` 等 API 一致；不要按原始 IRENE 的 PyTorch 1.8 环境运行当前代码。

## 16. 建议的新成员第一天清单

- [ ] 确认从 `cmr_tmp` 包根目录可以 import 项目。
- [ ] 确认 split、年龄 CSV 和四个 token 目录存在。
- [ ] 抽样验证四种 token 的 shape、dtype 和 finite values。
- [ ] 扫描四个 raw split 的 NPZ 路径和 T/Z 最小要求。
- [ ] 确认 ViTa、DINO、`multimode_train` 和四个单模态 checkpoint。
- [ ] 阅读 `startup_metrics.json`，确认当前实验的 shape、LIA 和训练策略。
- [ ] 先做 batch 1、LIA disabled 的速度/显存诊断。
- [ ] 再恢复正式 batch、LIA 和 deep supervision。
- [ ] 首个 epoch 后核对 train/val N、MAE、R²、LIA valid subjects 和学习率。
- [ ] 评估后确认 predictions 行数等于预期 test subject 数且 eid 唯一。

完成以上检查后，才应将训练结果用于正式对比或论文结论。
