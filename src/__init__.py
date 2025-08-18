# Source Generated with Decompyle++
# File: __init__.cpython-311.pyc (Python 3.11)

'''EEG 癫痫发作预测项目源代码包。

本包包含：
- 数据读取与预处理：`edf_reader`, `tse_parser`, `features`, `dataset`
- 训练与推理：`train`, `predict`, `model`
- 实验工具：`scan_thresholds`, `compare`, `utils`

约定：
- 统一使用 `configs/config.yaml` 配置训练与推理参数；
- 帧级分类以 `"bckg"` 为背景类名称；
- 尽量保持所有 I/O 与中间结果可复现（如 `data_cache/`、`outputs*/`）。
'''
__all__ = [
    'compare',
    'dataset',
    'edf_reader',
    'features',
    'model',
    'predict',
    'scan_thresholds',
    'train',
    'tse_parser',
    'utils']
