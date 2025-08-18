# EEG-Epilepsy-Prediction

多通道 EEG 癫痫发作检测与分型。端到端流程：EDF 读取 → 频带/时域特征 → BiLSTM 帧级多分类（含 `bckg`）→ 事件级后处理（平滑/确认/冷却/最小时长）→ 指标评估与阈值网格搜索（FA/h 约束，自动写回配置）。

---

## 一、整体架构与思路原理

- 数据层（`src/edf_reader.py`）
  - 使用 `pyedflib` 逐通道读取 EDF；去直流（零均值）；可选工频陷波（50/60Hz），可选带通/高通/低通；统一重采样到 `resample_hz`；将不同长度的通道裁剪到相同最小长度并堆叠为 `[C, N]`。
- 特征层（`src/features.py`）
  - 以滑窗参数 `window_sec/hop_sec` 计算 Welch PSD，聚合频带功率（delta/theta/alpha/beta/gamma），对通道做 {mean, std}；再加入宽频能量 {mean, std} 与时域 RMS {mean, std} → 得到逐帧特征 `[T, F]` 和帧中心时间 `centers`。
- 模型层（`src/model.py`）
  - BiLSTM（双向）提取时序上下文，线性头输出帧级 logits `[B,T,C]`；损失为交叉熵（忽略标签值 -100）。
  - 训练支持学习率调度（Cosine/OneCycleLR）、增强（Mixup/SpecAugment/噪声）、梯度裁剪，提升泛化与稳定性。
- 后处理（`src/postprocess.py`）
  - 基于 `1 - p(bckg)`：平滑（移动平均）→ 阈值二值化 → 连续确认窗（confirm）→ 相邻片段冷却时间合并（cooldown）→ 最小事件时长过滤；片段内按概率和（或最大）选主类与置信。
- 评估与阈值（`src/metrics.py`、`src/scan_thresholds.py`、`src/eval.py`）
  - 事件级 IoU 匹配：计算 P/R/F1、FA/h、起止延迟；阈值网格搜索在 FA/h 约束下选最佳，并自动写回 `configs/config.yaml`。

流程：

```
EDF files
  └─> edf_reader (filter / notch / resample / align)
        └─> features (Welch PSD bands + RMS)
              └─> BiLSTM frame classifier
                    ├─> postprocess (smooth / confirm / cooldown / min_dur) -> events
                    ├─> metrics (PR/F1, FA/h, latencies)
                    └─> threshold grid search (constraints + writeback)
```

---

## 二、安装与环境

- Python 3.11+（建议虚拟环境 conda/venv）
- 安装依赖（按需选择 CPU 或 CUDA 的 PyTorch 轮子）：

```
# 方式 A：CPU（示例）
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt

# 方式 B：CUDA 12.1（示例）
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install -r requirements.txt

# 验证
python -c "import torch;print('cuda_available=', torch.cuda.is_available())"
```

- Windows 终端若中文乱码：使用 PowerShell，并确保文件保存为 UTF-8。

---

## 三、数据组织与缓存

- 数据集参考：TUSZ（Temple University Hospital Seizure Corpus，见官网文档）。
  - 官方主页：`https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml`

- 将 EDF 与同名 TSE 放入 `data/Dataset_train_dev/`，TSE 行示例：
  - `0.0000 36.8868 bckg 1.0000`（开始秒 结束秒 标签 置信；置信可缺省）
- 目录示例：

```
data/
  Dataset_train_dev/
    00000001_s001_t000.edf
    00000001_s001_t000.tse
    ...
```

- 首次运行会在 `data_cache/` 生成 `.npz` 特征缓存，加速后续流程。
- `.gitignore` 已忽略：`data/`, `data_cache/`, `outputs/`, `__pycache__/`。

---

## 四、快速上手（逐步操作）

1) 准备数据

- 按上节放好 EDF/TSE；确保 `pyedflib` 可正常读取 EDF。

2) 配置

- 编辑 `configs/config.yaml` 的 `train`、`postprocess`、`labels` 三节；或使用命令行参数覆盖。
  - 如需多类别检测/分型，请提供两张 Excel（示例字段：规范名/别名），训练会自动生成并使用 `outputs/labels.json`；
  - 训练前会进行“类别一致性”前置校验，`.tse` 中的标签必须被 Excel 定义的标签/别名覆盖。

3) 训练（OneCycleLR + Mixup 示例）

```
python -m src.train --config configs/config.yaml \
  --scheduler onecycle --mixup_alpha 0.2
```

- TensorBoard：`tensorboard --logdir outputs/tb`
- 最佳模型：`outputs/best.pt`

4) 阈值网格（FA/h 约束 + 写回配置）

```
python -m src.scan_thresholds \
  --data_dir data/Dataset_train_dev --cache_dir data_cache \
  --probs 0.6 0.7 0.8 0.9 --smooth 0.0 0.25 \
  --confirm 1 2 3 --cooldown 0.0 0.5 1.0 \
  --min_duration 0.0 --max_fa_per_hour 2.0 \
  --out outputs/threshold_grid.json --write_config configs/config.yaml
```

- 输出：`outputs/threshold_grid.json`；将最佳阈值写回 `configs/config.yaml:postprocess`。

5) 评估（从 checkpoint）

```
python -m src.eval --config configs/config.yaml --checkpoint outputs/best.pt \
  --labels_json outputs/labels.json
```

- 输出：
  - `outputs/eval_summary.json`（多 IoU 全局指标）
  - `outputs/eval_records.csv`（逐记录 TP/FP/FN 与起止延迟）

6) 单文件检测（推理）

```
python -m src.predict --edf path/to/file.edf --checkpoint outputs/best.pt \
  --config configs/config.yaml --labels_json outputs/labels.json \
  --out outputs/pred_events.json
```

- 输出：是否存在癫痫事件、段数、每段类型与起止时间。

7) 复现与缓存

- 随机种子：`--seed`（训练脚本）
- 重算特征：删除 `data_cache/*.npz` 后再次运行。

---

## 五、配置详解（摘录）

`configs/config.yaml`

```
train:
  # 含有 EDF/TSE 成对文件的数据根目录
  data_dir: <path_to_dataset>  # 例：data/Dataset_train_dev
  # 特征缓存目录（存放 .npz，加速复用）
  cache_dir: <path_to_cache>   # 例：data_cache
  # 特征滑窗长度（秒）
  window_sec: 2.0
  # 特征滑窗步长（秒）
  hop_sec: 0.25
  # 读取后统一重采样的频率（Hz）
  resample_hz: 256.0
  # 预处理带通范围（Hz）；某端置为 null 可退化为高通/低通或不启用
  bandpass: [0.5, 45.0]
  # 工频陷波（50 或 60）；置为 null 不启用
  notch_hz: 50.0
  # 背景类名称（需与 .tse 或别名映射一致）
  bg_label: bckg
  # 批大小（根据显存/内存调整）
  batch_size: 4
  # 训练轮次（更大数据集可适当增大）
  epochs: 5
  # 按病人划分的验证/测试比例
  val_ratio: 0.2
  test_ratio: 0.0
  # 随机种子（复现）
  seed: 42
  # 输出目录（日志/权重/TensorBoard）
  out_dir: outputs

  # 学习率调度与数据增强
  # 建议：小中型数据集可选 onecycle；也可用 none/cosine
  scheduler: onecycle  # 可选：none|cosine|onecycle
  max_lr: 0.001
  # 梯度裁剪阈值（0 表示不裁剪）
  clip_grad: 0.0
  # Mixup 强度（>0 开启帧级软标签混合）
  mixup_alpha: 0.0
  # SpecAugment 遮挡（0 表示不启用）
  spec_time_mask_ratio: 0.0
  spec_time_masks: 0
  spec_feat_mask_ratio: 0.0
  spec_feat_masks: 0
  # 特征级高斯噪声强度
  aug_noise_std: 0.0

postprocess:
  # 基于 1 - p(background) 的判定阈值
  prob: 0.8
  # 概率平滑窗口（秒）
  smooth: 0.25
  # 事件确认所需的连续帧数
  confirm: 2
  # 冷却合并时间（秒，同类相邻事件在此间隔内合并）
  cooldown: 0.5
  # 最短事件时长（秒，低于此阈值丢弃）
  min_duration: 0.0

labels:
  # 背景类名称（需与 train.bg_label 保持一致）
  background: bckg
  # Excel 表（取第一个 sheet）：每行前两个非空单元格视为 (label, alias)
  excel_types: <path_to_types.xlsx>
  excel_periods: <path_to_periods.xlsx>
  # 训练/评估/推理共用的标签与别名导出文件
  json_out: <path_to_labels.json>
```

要点说明：

- `window_sec/hop_sec` 控制时间分辨率与计算量；`resample_hz` 建议与数据接近的频率（如 256Hz）。
- `bandpass/notch_hz` 用于抑制基线漂移与工频噪声；根据实验环境（50/60Hz）调整。
- 调度器：`onecycle` 在中小数据集上通常更稳定；`cosine` 简洁有效。
- 增强：`mixup_alpha>0` 开启帧级软标签混合；SpecAugment 用于时间/特征维遮挡；`aug_noise_std` 轻度高斯噪声。
- 后处理：`prob` 越高越保守（减少 FP）；`confirm/cooldown/min_duration` 控制事件碎片化与误报。

---

## 六、特征与模型（更细节）

- 频带：delta(0.5–4)、theta(4–8)、alpha(8–13)、beta(13–30)、gamma(30–45)。
- 每帧特征：
  - 5 个频带功率 {mean, std} → 10 维
  - 宽频能量 {mean, std} → 2 维
  - 时域 RMS {mean, std} → 2 维
  - 合计约 14 维（可扩展更多统计或比值特征）。
- BiLSTM：输入 `[B, T, F]`，输出 `[B, T, C]`；标签 -100 作为忽略项。
- 增强：
  - Mixup：同一 batch 内按 Beta(α,α) 线性混合并生成软目标。
  - SpecAugment：随机时间段或特征频段置零，模拟遮挡与丢失。
  - 噪声扰动：对特征施加小幅高斯噪声。

---

## 七、指标与后处理

- IoU 匹配：pred 与 gt 片段的区间 IoU≥阈值且类别一致 → TP；未匹配的 pred 为 FP，未匹配的 gt 为 FN。
- P/R/F1：按累计 TP/FP/FN 计算；支持多 IoU（`--ious`）。
- FA/h：`FP / 总小时`，用于误报警率控制（阈值网格时可用 `--max_fa_per_hour` 约束）。
- 起止延迟：匹配对的 |Δonset| / |Δoffset| 平均。
- 后处理流水：平滑 → 二值化 → 确认窗过滤 → 冷却合并 → 最小时长过滤；片段类别以帧概率和（或最大）取主类。

---

## 八、项目结构

- `src/edf_reader.py`：EDF 读取、滤波/陷波、重采样、对齐
- `src/features.py`：谱功率与时域特征
- `src/dataset.py`：数据集与 `.npz` 缓存
- `src/model.py`：`BiLSTMClassifier`
- `src/postprocess.py`：平滑/确认/冷却/时长过滤
- `src/metrics.py`：IoU 匹配、P/R/F1、FA/h、延迟
- `src/scan_thresholds.py`：阈值网格、FA/h 约束、写回配置
- `src/eval.py`：从 checkpoint 评估，输出 JSON/CSV
- `src/train.py`：训练（配置、日志、TB、scheduler、augment、checkpoint）
- `configs/config.yaml`：配置模板
- `data/`、`data_cache/`、`outputs/`：数据/缓存/结果目录（已忽略）

---

## 九、常见问题（FAQ）

- SciPy/pyEDFlib 安装失败：建议在 conda 环境下安装相应二进制包；或使用与 Python 版本匹配的 whl。
- 训练显存不足：降低 `batch_size`；增大 `hop_sec` 降低帧数；或下调 `hidden_dim`。
- 指标异常：
  - 检查 `bg_label` 是否与数据一致；
  - 确认 TSE 解析与时间单位；
  - 查验后处理阈值是否已写回并被评估脚本正确加载。
- 缓存冲突：修改特征/滤波参数后建议删除旧的 `data_cache/*.npz` 以免混用。

---

## 十、许可与致谢

- 许可：见根目录 `LICENSE`。
- 致谢：感谢开源社区（pyEDFlib、SciPy、PyTorch、TensorBoard 等）提供的生态支持。
 - 数据集：感谢 Temple University Hospital Seizure Corpus（TUSZ）提供的数据与标注，参考其[官方主页](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)。
