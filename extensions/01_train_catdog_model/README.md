# 扩展案例 1：猫狗分类模型训练

> 类型：`PY` ｜ 验证状态：待验证 ｜ 对应微课：[`microcourses/01_train_sin_model`](../../microcourses/01_train_sin_model/)

## 摘要

微课最小案例 1 用 sin 模型走通了"数据 → 训练 → 导出 ONNX"的最小流程；本扩展案例把同一套流程用到**真实图片**上：训练一个能区分猫和狗的分类模型，这是 Memory Pendant"看见并识别有限类别"核心能力的雏形。

和最小案例 1 相比有四个升级：**数据**变成磁盘上的真实图片（`ImageFolder` 按目录结构读取）；**模型**从 3 层小 MLP 换成 **MobileNetV2**（专为手机/嵌入式设备设计的轻量 CNN）；**预处理**出现了——图片统一缩放到 224×224 并归一化，这套参数将来要原样搬到 ESP32 端；**指标**除 loss 外新增验证准确率 `val_acc`。

产物：每轮 train/val loss、准确率、`metrics.csv`、loss 曲线、`.pth` 权重和供扩展案例 2 量化用的 `.onnx`，目标芯片 `esp32s3`。**只在电脑上运行，不需要开发板**；使用公开猫狗数据集。

# Quick Start

**预备知识**

- **已完成微课最小案例 1**：训练循环、loss、ONNX 导出等概念相同，不懂的名词先查最小案例 1 术语表；
- **conda 环境** **`esp32s3`**：依赖比最小案例 1 多 `torchvision`（图片数据集和模型库）；
- **准备公开猫狗数据集**：如 Kaggle "Dogs vs. Cats"，**记录名称、来源和许可证**（验收要求）；数据集通常几百 MB，CPU 训练明显慢于最小案例 1，有 NVIDIA GPU 会自动启用。

**数据准备（重要，先做这一步）**

`ImageFolder` 的规矩：数据根目录下每个子目录是一个类别，目录名就是类别名。本案例强制要求恰好两个小写目录 `cat` 和 `dog`（脚本会检查，不符合直接报错）。

下面提供两个可直接 `wget` 的公开镜像源，建议教学使用第一个**小型数据集**（已分 train/valid，体积小）：

**选项 A：小型数据集（推荐，约 2400 张，已分 train/valid）**

来源：`https://github.com/dl7days/datasets/raw/master/cats-dogs-data.zip`

```bash
wget -O cats-dogs-data.zip 'https://github.com/dl7days/datasets/raw/master/cats-dogs-data.zip' \
    && unzip cats-dogs-data.zip
```

解压后结构：

```text
cats-dogs-data/
├── train/
│   ├── cat/
│   └── dog/
└── valid/
    ├── cat/
    └── dog/
```

- 2000 张训练图 + 400 张验证图；
- 适合直接演示；脚本支持用 `--data-dir cats-dogs-data/train --valid-data-dir cats-dogs-data/valid` 运行，不再内部随机划分。

**选项 B：完整数据集（约 25000 张，已分目录）**

来源：`https://github.com/laxmimerit/dog-cat-full-dataset/archive/refs/heads/master.zip`

```bash
wget -O cats-dogs-full.zip 'https://github.com/laxmimerit/dog-cat-full-dataset/archive/refs/heads/master.zip' \
    && unzip cats-dogs-full.zip
```

完整版路径为 `dog-cat-full-dataset-master/data/`，子目录为 `cats`/`dogs`，如需和本脚本对接需先整理成 `cat`/`dog` 名称并统一 train/valid 结构，也可只作为延伸阅读材料。

**选项 C：自制 train/valid 结构**

如果你有自己的猫狗图片，也按 `train/cat`、`train/dog`、`valid/cat`、`valid/dog` 放好：

```text
data/catdog/
├── train/
│   ├── cat/   (cat.0.jpg, cat.1.jpg, ...)
│   └── dog/   (dog.0.jpg, dog.1.jpg, ...)
└── valid/
    ├── cat/
    └── dog/
```

- ImageFolder 按**字典序**给类别编号，所以 `cat`=0、`dog`=1；脚本显式检查 `dataset.classes != ["cat", "dog"]` 就抛错，防止编号悄悄变化；
- 训练、验证目录都至少要有一点点图；要训出可用的分类器，每类请尽量多放图片；jpg/png 都可以，损坏文件先清理；
- 数据集不要提交到代码仓库，只放本地 `data/catdog/`。

**第 1 步：激活 conda 环境**（成功后提示符出现 `(esp32s3)`）

```bash
conda activate esp32s3
```

**第 2 步：进入案例目录**（`ls` 应能看到 `train_catdog.py` 和 `requirements.txt`）

```bash
cd extensions/01_train_catdog_model
```

**第 3 步：安装依赖**（torch/torchvision 体积大，首次约 5\~15 分钟）

```bash
python -m pip install -r requirements.txt
```

**第 4 步：放好数据并自查**

```bash
ls cats-dogs-data/train
ls cats-dogs-data/train/cat | head -3
ls cats-dogs-data/valid/dog | head -3
```

**第 5 步：运行训练**

脚本会先检查类别必须是 `['cat', 'dog']`、图片数量足够，不通过直接抛 `ValueError`；每轮打印一行 `epoch=N train_loss=... val_loss=... val_acc=...`；GPU 下每轮几十秒到几分钟，**纯 CPU 明显更慢**，想先验证流程可用小子集试跑。

```bash
python train_catdog.py \
    --data-dir cats-dogs-data/train \
    --valid-data-dir cats-dogs-data/valid \
    --epochs 3 --output-dir outputs
```

**第 6 步：检查产物**（`ls` 应看到 5 个文件：`catdog_mobilenet_v2.pth`、`catdog_mobilenet_v2.onnx`、`labels.txt`、`metrics.csv`、`loss_curve.png`；`labels.txt` 应为两行 `cat`、`dog`，顺序即输出编号 0/1 的含义；ONNX 是扩展案例 2 的输入，妥善保留）

```bash
ls outputs
head -4 outputs/metrics.csv
cat outputs/labels.txt
```

## 预期现象

标准输出**格式**如下（格式示例，具体数值以本机运行为准；本案例验证状态为"待验证"，准确率不预设固定值）：

```text
epoch=1 train_loss=0.xxxxx val_loss=0.xxxxx val_acc=0.xxxx
...（每轮一行，共 --epochs 行）...
```

- 每轮一行；ONNX 导出是静默完成的，以 `outputs/` 中 5 个文件是否存在为准；从零训练（`weights=None`）且数据/轮数有限时，`val_acc` 可能不高（0.5\~0.7 都属常见），重点是把它与数据规模、划分、seed、训练设备一起如实记录；
- 失败时：类别顺序错误查目录名（不要手改 `labels.txt` 掩盖）；内存不足降 batch size 并记录改动；ONNX 导出失败保留版本信息，不把 PTH 直接交给端侧工程。

**常见问题排查**

| 现象                                                      | 原因                               | 解决办法                                        |
| ------------------------------------------------------- | -------------------------------- | ------------------------------------------- |
| `ValueError: expected classes ['cat', 'dog']`           | 有多余子目录、大小写不对或混入 `__MACOSX` 等隐藏目录 | 严格整理成两个小写目录，删掉多余目录                          |
| `ValueError: dataset must contain at least four images` | 图片没放对位置或太少                       | 用 `ls data/catdog/cat \| wc -l` 分别统计两类数量    |
| `ModuleNotFoundError: No module named 'torchvision'` 等  | 依赖没装全或没激活环境                      | 激活 `esp32s3` 环境后重装 requirements.txt         |
| 训练极慢、CPU 占满                                             | 没有 GPU，MobileNetV2 在 CPU 上本来就慢   | 属正常；减少图片或 epoch 先跑通，记录实际训练设备                |
| `RuntimeError: ... out of memory`                       | batch size 相对硬件太大                | 加 `--batch-size 8` 甚至 `4` 重跑并记录             |
| `val_acc` 一直在 0.5 附近不动                                  | 从零训练 + 数据太少或 epoch 太少            | 不是 bug：加图片、加大 `--epochs`，或按配置说明改用预训练权重，如实记录 |

## 学习目标

- 按 ImageFolder 规范准备 cat/dog 两类数据，说出为什么 cat=0、dog=1；
- 从每轮输出读出 train\_loss、val\_loss、val\_acc 并解释趋势；
- 解释 224×224 缩放和 ImageNet 归一化这套预处理为什么必须与端侧一致；
- 检查 `labels.txt`、`metrics.csv`、`.pth`、`.onnx` 四类产物的内容和用途，说出 `weights=None`（从零训练）的含义及对准确率的现实预期。

## 原理讲解

**MobileNetV2：为"小设备"而生的 CNN**

MLP 把每个输入和输出全连起来，处理 224×224×3 的图片（15 万个像素值）参数量会爆炸。CNN 换了思路：用小窗口（卷积核）在图片上滑动，提取边缘、纹理等局部特征，再逐层组合成"耳朵、眼睛、轮廓"——参数少、效果好，是视觉模型的主流结构。MobileNetV2 是 Google 为手机设计的轻量 CNN，用"深度可分离卷积"大幅减少计算量，常被选作 ESP32-S3 这类小芯片的视觉模型起点。

诚实说明：脚本里 `models.mobilenet_v2(weights=None)` 表示**从零随机初始化训练，不加载官方预训练权重**。图片少、只跑默认 3 个 epoch 时，验证准确率可能只有 50%\~70%（相当于瞎猜到略好于瞎猜）——这是正常现象，不是代码坏了。本案例的验收重点是**流程正确、指标可记录**，而不是刷准确率。

![训练、量化到端侧部署流程图（AI 生成示意）](../../docs/assets/ai_generated/vision_train_quant_deploy.png)

> 图：本扩展处于视觉模型链路的训练阶段，产物是供量化和部署使用的 ONNX 模型。类别顺序和预处理参数以脚本为准。

**预处理：Resize + ToTensor + Normalize**

每张图片进模型前都要走同一条流水线：

1. `Resize((224, 224))`：统一缩放到 MobileNetV2 的标准输入尺寸；
2. `ToTensor()`：像素从 0\~255 整数变成 0\~1 浮点张量，形状 3×224×224；
3. `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`：按 ImageNet 统计的均值/标准差归一化。

**Normalize（归一化）是什么？**

归一化，大白话就是把原始像素值"拉到同一个尺度"，让模型看得更舒服、训练更稳定。

`ToTensor()` 已经把像素从 `0\~255` 变成了 `0\~1`，但不同图片的亮度、颜色分布还是可能相差很大。Normalize 的公式是：

```text
pixel = (pixel - mean) / std
```

也就是先减去均值，再除以标准差，把数值大致拉到以 0 为中心、标准差为 1 的分布。

- `mean=[0.485, 0.456, 0.406]`：ImageNet 上百万张图片在 R、G、B 三个通道的平均颜色；
- `std=[0.229, 0.224, 0.225]`：对应通道的标准差。

举个例子：某像素红色通道值是 `0.8`（已经是 `0~1` 之间），Normalize 后：

```text
(0.8 - 0.485) / 0.229 ≈ 1.38
```

数值从 `0~1` 变成了大约 `-2~2` 的范围，模型对这种分布更"熟悉"，训练更容易收敛。

**这套参数是"合同"**：将来 ESP32 端对摄像头图片的预处理必须和这里一模一样，否则模型看到的输入分布和训练时不同，识别结果会莫名其妙地变差。

**分类输出、loss 与数据划分**

最后一层被改成 `nn.Linear(model.last_channel, 2)`：输出 2 个分数（logits），第 0 个对应 `cat`、第 1 个对应 `dog`，谁大判谁（`argmax`）。分类任务不用 MSE，标配是**交叉熵 loss（CrossEntropyLoss）**：衡量预测概率分布和真实类别的差距。数据按固定 seed 约 80% 训练、20% 验证，并做两个保底：训练集至少 2 张；验证集取整后为 0 就从训练集挪 1 张——保证小数据集下流程也能完整走通。

## 整体流程图

```text
         +------------------------------------------+
         |  公开猫狗数据集（已分 train / valid）        |
         |  cats-dogs-data/train/cat + dog           |
         |  cats-dogs-data/valid/cat + dog           |
         +--------------------+---------------------+
                              |
                              v
         +------------------------------------------+
         |  预处理：Resize(224,224)                 |
         |           ToTensor()                     |
         |           Normalize(ImageNet 均值/标准差) |
         +--------------------+---------------------+
                              |
                              v
         +------------------------------------------+
         |  模型：MobileNetV2(weights=None)         |
         |        分类头改为 2 类                   |
         |  优化：Adam(lr=1e-4)                     |
         |  损失：CrossEntropyLoss                 |
         +--------------------+---------------------+
                              |
              +--------------+--------------+
              |                             |
              v                             v
   +-----------------------+     +-------------------------+
   |  训练（train）         |     |  验证（valid）           |
   |  更新权重              |     |  计算 val_loss / val_acc |
   +-----------------------+     +-------------------------+
                              |
                              v
         +------------------------------------------+
         |  产物                                      |
         |  catdog_mobilenet_v2.pth（PyTorch 权重）  |
         |  catdog_mobilenet_v2.onnx（扩展案例 2 输入）|
         |  labels.txt / metrics.csv / loss_curve.png |
         +------------------------------------------+
```

## 关键代码解析

片段来自 [`train_catdog.py`](train_catdog.py)。

**1. 预处理流水线与防呆检查**

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # 统一尺寸
    transforms.ToTensor(),                  # 0~255 → 0~1，3×224×224
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),  # ImageNet 统计值
])
dataset = datasets.ImageFolder(args.data_dir, transform=transform)
if dataset.classes != ["cat", "dog"]:       # 类别名+顺序的硬性检查
    raise ValueError(...)
if len(dataset) < 4:                        # 数据量下限
    raise ValueError(...)
```

**2. 模型改造：1000 类头换成 2 类**

```python
model = models.mobilenet_v2(weights=None)   # 从零训练，不加载预训练权重
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 输出改为 2 类
criterion = nn.CrossEntropyLoss()           # 分类任务标准 loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

只替换最后一层全连接，其余结构原样保留。评估时 `logits.argmax(1)` 即"两个分数谁大判谁"，accuracy = 判对张数 ÷ 总张数。

**3. ONNX 导出：固定 batch=1、静态 shape**

```python
model.eval()
dummy = torch.randn(1, 3, 224, 224)   # 假输入：1 张 3 通道 224×224 图片
torch.onnx.export(
    model.cpu(), dummy, output_dir / "catdog_mobilenet_v2.onnx",
    opset_version=12, input_names=["input"], output_names=["output"], dynamo=False
)
```

与最小案例 1 同样的约定（batch=1、静态 shape、opset 12）；`model.cpu()` 避免把 CUDA 信息带进 ONNX。与最小案例 1 不同，本脚本没有 onnx.checker/onnxsim 检查，导出成功与否以文件生成、扩展案例 2 能否读取为准。

## 关键文件说明

| 文件/目录                                       | 职责                                     | 类型             |
| ------------------------------------------- | -------------------------------------- | -------------- |
| [`train_catdog.py`](train_catdog.py)        | 训练、验证和导出入口                             | 课程代码           |
| [`requirements.txt`](requirements.txt)      | Python 依赖（含 torchvision，不含 esp-ppq）    | 配置             |
| `data/catdog/`                              | ImageFolder 输入（cat/dog 两个子目录）          | 外部数据，需记录来源与许可证 |
| `outputs/catdog_mobilenet_v2.pth` / `.onnx` | PyTorch 权重 / 扩展案例 2 的量化输入              | 生成物            |
| `outputs/labels.txt`                        | 类别编号对照（第 0 行 cat，第 1 行 dog）            | 生成物            |
| `outputs/metrics.csv`                       | 每轮 train\_loss/val\_loss/val\_accuracy | 生成物            |
| `outputs/loss_curve.png`                    | 训练/验证 loss 曲线                          | 生成物            |

## 配置说明

| 修改位置                                 | 配置项                                      | 现象变化                                                 |
| ------------------------------------ | ---------------------------------------- | ---------------------------------------------------- |
| 命令行                                  | `--data-dir`（必填）                         | 数据来源；必须保持 cat/dog 两类结构                               |
| 命令行                                  | `--epochs`（默认 3） / `--batch-size`（默认 16） | 训练时间与拟合程度 / 吞吐与内存，内存不足调小                             |
| 命令行                                  | `--seed`（默认 42） / `--output-dir`         | 固定划分便于复现 / 产物目录                                      |
| [`train_catdog.py`](train_catdog.py) | `Resize((224,224))` 和 `Normalize` 统计值    | 必须与模型和端侧一致，改动要两边同步                                   |
| [`train_catdog.py`](train_catdog.py) | `weights=None`                           | 改为 `"IMAGENET1K_V1"` 可加载预训练权重（需联网），准确率提升明显，但必须记录这一变化 |

## 术语小表

| 术语                    | 解释                                                   |
| --------------------- | ---------------------------------------------------- |
| CNN                   | 卷积神经网络，用小窗口在图片上滑动提取局部特征，图片任务的主力结构                    |
| ImageFolder           | PyTorch 数据集接口，按"一个子目录=一个类别"自动读图配标签                   |
| MobileNetV2           | 面向移动/端侧场景的轻量 CNN，本案例的模型骨架                            |
| 预训练权重                 | 别人在大数据集上训好的参数；本脚本 `weights=None` 表示不用、从零训            |
| 分类头 / logits / argmax | 网络最后输出各类别分数的那一层（本例换成输出 2 个分数）/ 原始分数 / "取最大值编号"得到最终判断 |
| 交叉熵 loss              | 分类任务的标准损失函数，衡量预测概率和真实类别的差距                           |
| val\_acc              | 验证集准确率 = 判对样本数 ÷ 总样本数                                |
| Normalize（归一化）        | 按均值/标准差把像素值拉到统一分布，本例用 ImageNet 统计值                   |
| 字典序                   | 像词典一样按字母排序；ImageFolder 据此编号，所以 cat=0、dog=1           |

## 验证清单

- [ ] 数据类别严格为 `cat`、`dog`，记录来源/许可证、seed、epoch、batch size、训练设备（CPU/GPU）和两类图片数量；
- [ ] 脚本退出码为 0，每轮打印 `epoch=... train_loss=... val_loss=... val_acc=...`；
- [ ] `metrics.csv`、`loss_curve.png`、`labels.txt`、`.pth`、`.onnx` 五个文件均生成，`labels.txt` 为两行 `cat`、`dog`；
- [ ] 最终 val\_acc 与各轮 loss 填入实验记录，附数据规模说明；
- [ ] 不提交真实个人照片或未经许可的数据。

## 思考题与拓展挑战

**思考题**

1. 为什么类别目录顺序必须固定？

   \*\*参考答案：\*\*模型输出的第 0/1 维需要稳定映射到 cat/dog；顺序变了，预测数值就会被错误解释。
2. 本脚本 `weights=None` 从零训练。改用 ImageNet 预训练权重后，loss 和 val\_acc 曲线会有什么不同？

   \*\*参考答案：\*\*预训练权重已学会通用图像特征，只需微调分类头，因此 loss 起点更低、下降更快，同样 epoch 下 val\_acc 通常明显更高；代价是需联网下载，且必须在记录中注明配置变化。
3. 为什么 `Normalize` 用 ImageNet 统计值，而不是自己数据集算出来的？

   \*\*参考答案：\*\*这是 MobileNetV2 生态的通行约定，预训练权重和端侧示例都基于它；用自己的统计值也行，但训练端与部署端必须严格一致——一致性比"更准的统计值"更重要。

**拓展挑战**

- 在保持类别和输出格式不变的约束下，增加一个混淆矩阵产物；
- 设计一个公开、合规且不包含个人图像的端侧测试集。

