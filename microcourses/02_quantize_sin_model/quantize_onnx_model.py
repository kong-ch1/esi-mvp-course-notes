"""第 2 集微课最小案例：用 ESP-PPQ 将 sin 模型的 ONNX 量化为 .espdl。

前置：先运行 `python sin_model.py` 生成 outputs/sin_model.onnx 与 outputs/sin_model.pth。

产物（写入 outputs/）：
  - sin_model.espdl    ESP-DL 可直接加载的量化模型（int8，target=esp32s3）
  - sin_model.json     量化参数（scale/exponent 等）
  - sin_model.info     量化报告

脚本同时打印教案验收所需的两组对照数据：
  1. 量化前后精度对照：float model MSE vs quant model MSE（同一份数据）
  2. 模型大小对照：.onnx vs .espdl 的文件字节数

注意：export_test_values=True 会把一组测试输入/输出嵌入 .espdl，
第 2 集在板上调用 `model->test()` 自检时依赖它。
"""

import os
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parents[1] / "01_train_sin_model"
sys.path.insert(0, str(TRAIN_DIR))

import torch
from torch.utils.data import DataLoader, TensorDataset

from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.executor.torch import TorchExecutor
from sin_model import SinPredictor, generate_data

DEVICE = "cpu"  # 如需 GPU 请确认 CUDA 可用


def collate_fn(batch):
    # TensorDataset 迭代时返回 (x, y)，量化校准时只需要 x，不需要标签 y。
    return torch.stack([sample[0] for sample in batch]).to(DEVICE)


if __name__ == "__main__":
    torch.manual_seed(42)  # 校准数据可复现，且与训练数据同分布

    TRAIN_OUTPUT_DIR = TRAIN_DIR / "outputs"
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_MODEL_PATH = str(TRAIN_OUTPUT_DIR / "sin_model.onnx")
    PTH_MODEL_PATH = str(TRAIN_OUTPUT_DIR / "sin_model.pth")
    ESPDL_MODEL_PATH = str(OUTPUT_DIR / "sin_model.espdl")
    INPUT_SHAPE = [1, 1]  # 1 个输入特征，batch 必须为 1
    TARGET = "esp32s3"  # 量化目标类型，可选 'c', 'esp32s3', 'esp32p4'
    NUM_OF_BITS = 8  # 量化位数

    x, y = generate_data()
    # dataloader shuffle 必须设置为 False：
    # 计算量化误差时会多次遍历数据集，若 shuffle=True 会得到错误的量化误差。
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # 校准步数
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,  # 第 2 集 model->test() 板上自检依赖该选项
        verbose=1,
    )

    # ---- 对照 1：量化前后精度（同一份数据上的 MSE） ----
    criterion = torch.nn.MSELoss()

    # 浮点模型（量化前）
    model = SinPredictor()
    model.load_state_dict(torch.load(PTH_MODEL_PATH, weights_only=True))
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            loss += criterion(model(batch_x), batch_y)
    loss /= len(dataloader)
    print(f"float model MSE: {loss.item():.5f}")

    # 量化模型（量化后，用 PPQ 图模拟量化推理）
    executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)
    loss = 0.0
    for batch_x, batch_y in dataloader:
        y_pred = executor(batch_x)
        loss += criterion(y_pred[0], batch_y)
    loss /= len(dataloader)
    print(f"quant model MSE: {loss.item():.5f}")

    # ---- 对照 2：模型大小 ----
    onnx_size = os.path.getsize(ONNX_MODEL_PATH)
    espdl_size = os.path.getsize(ESPDL_MODEL_PATH)
    print(f"model size: {ONNX_MODEL_PATH} = {onnx_size} bytes")
    print(f"model size: {ESPDL_MODEL_PATH} = {espdl_size} bytes")
