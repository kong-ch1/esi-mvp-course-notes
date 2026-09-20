"""扩展案例 1：使用 ImageFolder 训练猫狗二分类模型并导出 ONNX。

用法示例：
    python train_catdog.py --data-dir cats-dogs-data/train \
        --valid-data-dir cats-dogs-data/valid --epochs 3
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def evaluate(model, loader, criterion, device):
    """在验证集上评估模型，返回平均 loss 和准确率。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 推理阶段不需要计算梯度
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--valid-data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 固定随机种子，让实验结果可复现
    torch.manual_seed(args.seed)
    # 自动选择 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图像预处理：缩放、转张量、按 ImageNet 统计值归一化
    # 训练端与部署端必须保持一致
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 加载训练集，ImageFolder 按子目录名自动分类
    train_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if train_dataset.classes != ["cat", "dog"]:
        raise ValueError(f"expected classes ['cat', 'dog'], got {train_dataset.classes}")

    # 验证集目录必须显式指定，脚本不再内部做 80/20 划分
    if args.valid_data_dir is None:
        raise ValueError("--valid-data-dir is required")

    # 加载验证集并检查类别顺序
    val_dataset = datasets.ImageFolder(args.valid_data_dir, transform=transform)
    if val_dataset.classes != ["cat", "dog"]:
        raise ValueError(f"expected classes ['cat', 'dog'], got {val_dataset.classes}")

    train_set = train_dataset
    val_set = val_dataset
    train_len = len(train_set)
    val_len = len(val_set)

    # 构建数据加载器：训练集打乱，验证集不打乱
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 加载 MobileNetV2 骨架，并把分类头改成 2 类（cat/dog）
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.to(device)

    # 分类任务常用交叉熵损失 + Adam 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    rows = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()          # 清空上一轮梯度
            loss = criterion(model(images), labels)  # 前向 + 计算损失
            loss.backward()                  # 反向传播算梯度
            optimizer.step()               # 更新权重
            train_loss += loss.item() * labels.size(0)
        train_loss /= train_len            # 平均训练损失

        # 每轮结束在验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        rows.append([epoch + 1, train_loss, val_loss, val_acc])
        print(f"epoch={epoch + 1} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}")

    # 保存模型权重和类别标签
    torch.save(model.state_dict(), output_dir / "catdog_mobilenet_v2.pth")
    (output_dir / "labels.txt").write_text("cat\ndog\n")

    # 把指标写入 CSV
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writerows(rows)

    # 绘制训练/验证 loss 曲线
    plt.plot([r[0] for r in rows], [r[1] for r in rows], label="train loss")
    plt.plot([r[0] for r in rows], [r[2] for r in rows], label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("cross entropy")
    plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 导出 ONNX 供后续端侧量化与部署使用
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.cpu(), dummy, output_dir / "catdog_mobilenet_v2.onnx",
        opset_version=12, input_names=["input"], output_names=["output"], dynamo=False
    )


if __name__ == "__main__":
    main()
