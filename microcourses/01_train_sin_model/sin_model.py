"""第 1 集微课最小案例：训练拟合正弦函数的 sin 模型，并导出 ONNX。

模型结构：3 层 MLP（1 -> 64 -> 64 -> 1，ReLU），输入 x ∈ [0, 2π]，输出 sin(x)。
产物（写入 outputs/）：
  - sin_model.pth      PyTorch 权重（量化前对照评估用）
  - sin_model.onnx     ONNX 模型（opset 12，batch=1，onnxsim 简化）
  - loss_history.csv   每个 epoch 的训练/验证 loss
  - loss_curve.png     训练/验证 loss 曲线
  - fit_curve.png      噪声真值 vs 模型预测对照图

用法：
  python sin_model.py                                # 零参数即可运行
  python sin_model.py --epochs 500 --seed 42         # 完整参数见 --help
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # 无显示器环境也能保存曲线

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

SEED = 42


class SinPredictor(nn.Module):
    def __init__(self):
        super(SinPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入层：1 个特征 (x)
        self.fc2 = nn.Linear(64, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, 1)  # 输出层：1 个特征 (y)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_data(num_samples=1000, noise=0.05):
    """生成 [0, 2π] 区间带噪声的正弦样本。量化脚本也复用该函数生成校准数据。"""
    x = torch.linspace(0, 2 * np.pi, num_samples)
    y = torch.sin(x) + noise * torch.randn(num_samples)
    return x.unsqueeze(1), y.unsqueeze(1)


def evaluate(model, dataloader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            total += criterion(model(batch_x), batch_y).item()
    return total / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="训练拟合正弦函数的 sin 模型并导出 ONNX")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数（默认 500）")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 42），固定后产物可复现")
    parser.add_argument("--num-samples", type=int, default=1000, help="样本数量（默认 1000）")
    parser.add_argument("--output-dir", type=str, default="outputs", help="产物输出目录（默认 outputs）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 数据与 train/val 划分（80/20）
    x, y = generate_data(num_samples=args.num_samples)
    dataset = TensorDataset(x, y)
    n_val = args.num_samples // 5
    n_train = args.num_samples - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = SinPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练：每轮记录 train loss 与 val loss
    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}], train loss: {train_loss:.5f}, val loss: {val_loss:.5f}")

    print(f"final train loss: {train_losses[-1]:.5f}, final val loss: {val_losses[-1]:.5f}")

    # loss 历史落盘（验收表格的数据源）
    csv_path = os.path.join(args.output_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, f"{tl:.6f}", f"{vl:.6f}"])
    print(f"saved: {csv_path}")

    # 训练/验证 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label="train loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_fig = os.path.join(args.output_dir, "loss_curve.png")
    plt.savefig(loss_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {loss_fig}")

    # 拟合对照图（噪声真值 vs 模型预测）
    model.eval()
    x_test, y_test = generate_data(num_samples=args.num_samples)
    with torch.no_grad():
        y_pred = model(x_test)
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.flatten().numpy(), y_test.flatten().numpy(), label="Noisy sine wave")
    plt.plot(x_test.flatten().numpy(), y_pred.flatten().numpy(), label="Predicted sine wave", color="r")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Original vs Predicted Sine Wave")
    plt.legend()
    plt.grid(True)
    fit_fig = os.path.join(args.output_dir, "fit_curve.png")
    plt.savefig(fit_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {fit_fig}")

    # 保存 PyTorch 权重
    pth_path = os.path.join(args.output_dir, "sin_model.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"saved: {pth_path}")

    # 导出 ONNX。esp-dl 只支持 batch=1、静态 shape；dynamo=False 使用经典导出器。
    onnx_path = os.path.join(args.output_dir, "sin_model.onnx")
    dummy_input = torch.randn([1, 1], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    onnx_model = onnx.load_model(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, onnx_path)
    print(f"saved: {onnx_path}")


if __name__ == "__main__":
    main()
