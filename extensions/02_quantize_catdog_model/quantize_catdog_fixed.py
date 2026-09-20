"""扩展案例 2：将猫狗 ONNX 分类模型量化为 ESPDL。

===============================================================================
⚠️ 这是老师 quantize_catdog.py 的【修正副本】，原文件一字未动。
   唯一改动：collate_fn 的契约写错了，导致脚本必崩。

【原版为什么崩】
  PPQ 内部的校准循环是：
      for data in calib_dataloader:
          data = self._collate_fn(data)          # <- 再调一次 collate_fn
          executor.forward(inputs=data, ...)
  也就是说 collate_fn 拿到的是 **DataLoader 已经拼好的 batch**，
  它的职责是「预处理」（转 dtype / 转 device），不是「把样本拼成 batch」。

  参考 PPQ 官方示例 esp_ppq/samples/quantize_onnx_model.py：
      def collate_fn(batch: torch.Tensor) -> torch.Tensor:
          return batch.to(DEVICE)

  而原版写的是：
      def collate_fn(batch):
          return torch.stack([sample[0] for sample in batch])
  它假设 batch 是「(图片, 标签) 样本列表」，于是：
      sample[0] 对图片张量 [8,3,224,224] 取到 [3,224,224]
      sample[0] 对标签张量 [8]            取到 0 维标量 []
      torch.stack 尺寸不一致 -> RuntimeError

【修正】取 batch 里的图片部分，统一转 float32。
  实测：原版崩溃，修正版输出 (8, 3, 224, 224) torch.float32。
===============================================================================
"""

import argparse
import os
from pathlib import Path

import torch
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def collate_fn(batch):
    # PPQ 传入的是 DataLoader 已拼好的 batch = [images, labels]（或单个张量）。
    # 模型只需要 images，丢掉 labels；再统一转 float32 供校准使用。
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    return batch.to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if dataset.classes != ["cat", "dog"]:
        raise ValueError(f"expected classes ['cat', 'dog'], got {dataset.classes}")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    espdl_path = output_dir / "catdog_mobilenet_v2.espdl"
    espdl_quantize_onnx(
        onnx_import_file=args.onnx,
        espdl_export_file=str(espdl_path),
        calib_dataloader=loader,
        calib_steps=min(32, len(loader)),
        input_shape=[1, 3, 224, 224],
        target="esp32s3",
        num_of_bits=8,
        collate_fn=collate_fn,
        device="cpu",
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
    )
    print(f"float model size: {os.path.getsize(args.onnx)} bytes")
    print(f"espdl model size: {espdl_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
