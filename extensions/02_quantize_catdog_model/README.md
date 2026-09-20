# 扩展案例 2：猫狗分类模型量化

> 类型：`PY` ｜ 验证状态：待验证 ｜ 前置：[`extensions/01_train_catdog_model`](../01_train_catdog_model/)

## 摘要

扩展案例 1 训出了猫狗模型的 ONNX，但它还是 float32 浮点模型——对 ESP32-S3 来说太大、太慢。本案例把微课最小案例 2 的量化流程搬到这个"真模型"上：用 **ESP-PPQ** 以 `esp32s3` 为目标做 **INT8 量化**，产出 ESP-DL 能加载的 **`.espdl`** 文件（后续端侧部署 LCD 实时猫狗分类等的模型来源），并打印量化前后的文件大小对照。**本案例只在电脑上运行，不需要开发板。**和微课最小案例 2 的三个不同点：

1. **校准数据变成真实图片**：用和训练完全相同的预处理读取 ImageFolder 里的猫狗图片；
2. **输入形状变成 `[1, 3, 224, 224]`**：一张 3 通道彩色图片；
3. **脚本只自动对照文件大小，不自动对照精度**：量化前后的分类准确率需按拓展挑战的方法在同一测试集上另行测量；实验报告里**不得把未执行的精度结果写成已实测**，也不得从文件大小推断精度。

## Quick Start

**预备知识**

- **已完成扩展案例 1**，需要两个上游产物：ONNX 模型 `../01_train_catdog_model/outputs/catdog_mobilenet_v2.onnx` 和校准图片 `../01_train_catdog_model/data/catdog/`（cat/dog 两个子目录）；**已完成微课最小案例 2**：量化概念（INT8、校准、scale/exponent、`.espdl`）最小案例 2 已讲过，这里不再从零解释；
- **注意 esp-ppq 依赖**：扩展 1 的 requirements.txt 里**没有**它（微课最小案例 1 的有 `esp-ppq==1.2.10`）。按顺序学过微课第 1、2 集的环境里应已装好；否则按 Quick Start 第 3 步单独安装；
- **时间预算**：量化 224×224 的 MobileNetV2 比 sin 小模型慢得多，CPU 上通常几分钟到几十分钟。


假设终端位于仓库根目录 `esi-mvp-code/`，且扩展案例 1 已完整跑完。

**第 1 步：激活 conda 环境**（成功后提示符出现 `(esp32s3)`）

```bash
conda activate esp32s3
```

**第 2 步：进入案例目录**（`ls` 应能看到 `quantize_catdog.py`；本目录没有自己的 requirements.txt）

```bash
cd extensions/02_quantize_catdog_model
```

**第 3 步：安装依赖（注意 esp-ppq）**（第一条装基础依赖，扩展 1 装过则秒完；**第二条不能省**：脚本第一行就 `from esp_ppq.api import espdl_quantize_onnx`，而扩展 1 的依赖清单不含 esp-ppq，学过微课最小案例 1 的环境里已有它、pip 会提示已满足；装完用 `python -c "import esp_ppq"` 验证）

```bash
python -m pip install -r ../01_train_catdog_model/requirements.txt
python -m pip install esp-ppq==1.2.10
```

**第 4 步：检查前置产物**（第一条无输出且退出码 0 表示 ONNX 存在；第二条应输出 `cat`、`dog` 两行——校准图片就是扩展 1 的训练数据目录；不满足先回扩展 1 补齐，不要手工造文件绕过）

```bash
test -f ../01_train_catdog_model/outputs/catdog_mobilenet_v2.onnx
ls ../01_train_catdog_model/data/catdog
```

**第 5 步：运行量化**（三个参数分别指定：被量化的模型、校准图片目录、产物输出目录；ESP-PPQ 会打印大量日志属正常；CPU 上通常几分钟到几十分钟，不要中途强杀）

```bash
python quantize_catdog.py \
  --onnx ../01_train_catdog_model/outputs/catdog_mobilenet_v2.onnx \
  --data-dir ../01_train_catdog_model/data/catdog \
  --output-dir outputs
```

**第 6 步：核对并记录**（确认 `catdog_mobilenet_v2.espdl` 已生成，ESP-PPQ 可能附带同名报告类文件；把终端最后两行字节数抄进实验记录并算缩小比例；若要填精度对照，必须用同一测试样本和相同标签映射，另存原始结果，不得凭文件大小推断精度）

```bash
ls outputs
```

## 预期现象

成功时终端末尾输出**格式**如下（格式示例，具体数值以本机运行为准；本案例验证状态为"待验证"）：

```text
...（ESP-PPQ 自身的校准/量化/导出日志，含量化误差报告）...
float model size: NNNNNNN bytes
espdl model size: NNNNNNN bytes
```

- 脚本自身只打印这两行，其余海量日志来自 ESP-PPQ；`espdl model size` 通常明显小于 `float model size`（INT8 存储收益），两个数字和比值一起记录；
- 脚本当前**未**自动输出量化前后分类准确率，这项数据应在扩展作业中补充记录，不得从文件大小推断精度；
- 失败时：ONNX 不存在先完成扩展 1；类别不对修正数据目录；校准或导出失败检查 shape、依赖版本和磁盘空间；`.espdl` 生成但精度异常时保留校准数据、模型 hash 和完整报告，暂停部署。

**常见问题排查**

| 现象 | 原因 | 解决办法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'esp_ppq'` | 扩展 1 的依赖清单不含 esp-ppq，环境里没装过 | 执行 `python -m pip install esp-ppq==1.2.10` |
| `ValueError: expected classes ['cat', 'dog']` | `--data-dir` 指错，或混入其他子目录/大小写不对 | 指向扩展 1 的 `data/catdog`，清理 `__MACOSX` 等多余目录 |
| `FileNotFoundError: ... catdog_mobilenet_v2.onnx` | 扩展 1 没跑完或路径写错 | 先完成扩展 1；`--onnx` 相对路径是相对**当前目录**的，逐字核对 |
| shape 不匹配类报错 | `input_shape` 与 ONNX 实际输入不一致 | 保持 `[1, 3, 224, 224]`；扩展 1 改过尺寸要两边同步 |
| 量化非常慢、疑似卡死 | MobileNetV2 在 CPU 上量化本来就慢，日志多不代表卡住 | 观察日志是否仍在滚动；预留几十分钟；确需提速可减小 `calib_steps`（须记录） |
| 内存不足/进程被杀 | 224×224 图片批量校准占内存较多 | 关闭其他大程序，把 `batch_size=8` 调小，记录改动 |
| `.espdl` 生成了但上板识别异常 | 校准预处理与训练/端侧不一致 | 逐字核对三处 transform；需要自检时改 `export_test_values=True` 重新量化；保留报告并暂停部署 |

## 学习目标

- 准备与训练类别、预处理完全一致的校准数据，并解释为什么必须一致；
- 运行量化脚本，确认 `.espdl` 正常生成；
- 从终端读出并记录 float（ONNX）与 ESPDL 两个文件的字节数；
- 解释本案例 `export_test_values=False` 与微课最小案例 2 `True` 的区别及影响；
- 设计"同一测试集上的量化前后精度对照"实验，诚实标注哪些实测、哪些未测。

## 原理讲解

**为什么校准数据要"一样的图片、一样的预处理"**

校准是拿代表性输入跑一遍浮点模型、统计每层数值范围，据此确定 INT8 刻度；刻度准不准，完全取决于这批输入像不像模型将来真正会遇到的输入。对猫狗模型来说，真实输入是"经过 `Resize(224,224) → ToTensor → Normalize(ImageNet 统计值)` 的猫狗图片张量"，所以脚本逐字复用扩展 1 的 transform，并读同一个 `data/catdog` 目录。如果校准忘了 Normalize 或用了别的尺寸，量化刻度就画错，上板后分类会莫名其妙地乱；脚本同样有类别防呆检查：`dataset.classes != ["cat", "dog"]` 直接抛 `ValueError`。

**量化调用：与最小案例 2 同构，三处不同**

校准 DataLoader 用 `batch_size=8`（量化内存压力比训练大）、`shuffle=False`（同最小案例 2：多次遍历必须可比较）、`calib_steps=min(32, len(loader))`（图片不足 32 批时用实际批数）。`espdl_quantize_onnx()` 参数结构与最小案例 2 一致，不同点：

| 参数 | 微课最小案例 2（sin） | 本案例（猫狗） | 说明 |
|---|---|---|---|
| `input_shape` | `[1, 1]` | `[1, 3, 224, 224]` | 1 张 3 通道 224×224 图片 |
| `export_test_values` | `True` | `False` | 不把测试值嵌入 `.espdl`；后续若需板上 `model->test()` 自检，改成 `True` 重新量化 |
| `calib_steps` | 固定 `32` | `min(32, len(loader))` | 数据不足时自动收敛到实际批数 |

其余相同：`target="esp32s3"`、`num_of_bits=8`、`device="cpu"`、`error_report=True`。

![浮点模型到 INT8 ESP-DL 模型的量化示意图（AI 生成示意）](../../docs/assets/ai_generated/float_to_int8_quantization.png)

> 图：代表性图片用于估计量化范围，量化后得到更适合端侧部署的整数模型。体积与精度关系是概念说明，实际结论以同一测试集测量为准。

**本脚本能证明什么、不能证明什么**

脚本最后只打印两行：`float model size`（ONNX 字节数）和 `espdl model size`（`.espdl` 字节数），能证明**存储收益**（INT8 通常明显更小）。但脚本**没有**在测试集上跑量化前后的分类推理，所以不能证明精度损失是多少——"文件更小"不等于"模型更好"。精度对照必须按拓展挑战第 1 条补充：用同一批测试图片、同样的标签映射（cat=0、dog=1），分别测浮点和量化的 Top-1 准确率后记录。

## 整体流程图

```text
扩展 1 产物：catdog_mobilenet_v2.onnx + data/catdog（ImageFolder）
             ↓ Resize(224,224)/ToTensor/Normalize（与训练逐字一致）
             ↓ DataLoader(batch_size=8, shuffle=False)
             ↓ espdl_quantize_onnx(target=esp32s3, num_of_bits=8,
                                   calib_steps=min(32, len(loader)),
                                   export_test_values=False)
outputs/catdog_mobilenet_v2.espdl → float/espdl model size 字节数对照
             ↓（需自行补充的实验）同一测试集上的量化前后 Top-1 精度记录
```

## 关键代码解析

片段来自 [`quantize_catdog.py`](quantize_catdog.py)。

**1. 预处理与类别检查：和训练脚本逐字一致**

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(args.data_dir, transform=transform)
if dataset.classes != ["cat", "dog"]:
    raise ValueError(f"expected classes ['cat', 'dog'], got {dataset.classes}")
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
```

这段 transform 与 `train_catdog.py` 完全一致是刻意复制："校准看到的分布 = 训练看到的分布 = 端侧将要看到的分布"。另有 `collate_fn` 只保留图片张量、丢掉标签，与微课最小案例 2 逻辑一致（那边多一句 `.to(DEVICE)`，本脚本固定 CPU 不需要）。

**2. 量化主调用**

```python
espdl_quantize_onnx(
    onnx_import_file=args.onnx,              # 扩展 1 导出的 ONNX
    espdl_export_file=str(espdl_path),       # outputs/catdog_mobilenet_v2.espdl
    calib_dataloader=loader,
    calib_steps=min(32, len(loader)),        # 数据不足 32 批时用实际批数
    input_shape=[1, 3, 224, 224],            # 必须与 ONNX 输入一致
    target="esp32s3",
    num_of_bits=8,                           # INT8
    collate_fn=collate_fn,
    device="cpu",
    error_report=True,                       # 打印量化误差报告
    export_test_values=False,                # 不嵌入板上自检用的测试值
    verbose=1,
)
```

**3. 文件大小对照**

```python
print(f"float model size: {os.path.getsize(args.onnx)} bytes")
print(f"espdl model size: {espdl_path.stat().st_size} bytes")   # 两种写法都是取文件字节数
```

## 关键文件说明

| 文件/目录 | 职责 | 类型 |
|---|---|---|
| [`quantize_catdog.py`](quantize_catdog.py) | 校准、量化和大小输出（本案例入口） | 课程代码 |
| [`../01_train_catdog_model/train_catdog.py`](../01_train_catdog_model/train_catdog.py) | 训练预处理和类别约定的来源（transform 需逐字一致） | 上游课程代码 |
| `../01_train_catdog_model/outputs/catdog_mobilenet_v2.onnx` | 量化输入 | 上游生成物 |
| `../01_train_catdog_model/data/catdog/` | 校准图片来源（ImageFolder 结构） | 外部数据 |
| `outputs/catdog_mobilenet_v2.espdl` | ESP-DL 部署模型 | 生成物 |

## 配置说明

配置写在 `quantize_catdog.py` 里或通过命令行传入，改后重跑：

| 修改位置 | 配置项 | 现象变化 |
|---|---|---|
| 命令行 | `--onnx` / `--data-dir` / `--output-dir` | 被量化的模型（须是扩展 1 产物）/ 校准图片目录（须保持 cat/dog 结构）/ 产物目录 |
| [`quantize_catdog.py`](quantize_catdog.py) | `input_shape=[1,3,224,224]` | 必须与 ONNX 输入一致，改错直接失败 |
| 同上 | `target="esp32s3"` / `num_of_bits=8` | 部署目标芯片 / 量化位宽，影响体积和误差 |
| 同上 | `calib_steps=min(32, len(loader))` | 校准覆盖范围，增大更稳但更耗时 |
| 同上 | `export_test_values=False` | 改为 `True` 会嵌入测试值，供板上 `model->test()` 自检 |
| 同上 | `device="cpu"` | 改 GPU 前须确认 CUDA 与 esp-ppq 兼容 |

## 术语小表

| 术语 | 解释 |
|---|---|
| 量化 | 把模型里的 float32 换成 INT8（8 位整数，只有 256 个档位），省存储、算得快，代价是少量精度 |
| 校准集 | 用来估计量化刻度的代表性输入；本例是与训练同预处理的猫狗图片 |
| 预处理一致性 | 校准、训练、端侧三方的 Resize/Normalize 必须逐字一致，否则刻度画错 |
| ESP-PPQ | 乐鑫的训练后量化工具，输入 ONNX、输出 `.espdl` |
| `espdl_quantize_onnx` | ESP-PPQ 核心 API，一个函数完成校准、量化、导出 |
| calib_steps | 校准使用的 batch 数；本例 `min(32, len(loader))` |
| `export_test_values` | 是否把测试值嵌入 `.espdl` 供板上自检；本例为 False |
| Top-1 准确率 | 概率最高的一个类别判对的比率；补充精度对照实验的指标 |
| 量化误差 | 量化表示与浮点表示之间的输出差异，error_report 会给出相关信息 |
| hash | 文件内容摘要（如 sha256），用于确认模型版本、归档实验记录 |

## 验证清单

- [ ] ONNX、校准数据和类别顺序来自同一次扩展 1 运行；
- [ ] `python -c "import esp_ppq"` 无报错；
- [ ] 记录目标芯片（esp32s3）、shape（[1,3,224,224]）、位宽（8）、校准步数和依赖版本；
- [ ] 保存 `float model size` / `espdl model size` 两行日志原文或字节数；
- [ ] 另用同一测试集记录量化前后精度，不用文件大小替代；未测项目明确标注"未实测"；
- [ ] 量化报告（终端日志/error report）和模型 hash 已归档。

## 思考题与拓展挑战

**思考题**

1. 为什么校准预处理必须和训练/端侧预处理一致？

   **参考答案：**量化范围由输入分布决定；预处理不一致会让校准范围与部署输入对不上，导致误差变大或分类偏移。

2. 为什么不能从 `.espdl` 文件更小就判断模型更好？

   **参考答案：**更小只说明存储成本可能降低，不能说明精度、延迟、内存和稳定性满足系统需求。

3. 本脚本 `export_test_values=False`，微课最小案例 2 是 `True`。后续要在板上用 `model->test()` 自检怎么办？

   **参考答案：**把 `export_test_values` 改为 `True` 重新量化，让测试值嵌入 `.espdl`，并记录配置变化；沿用 `False` 的产物无法通过自检。

**拓展挑战**

- 为脚本增加同一测试集上的 float/quant Top-1 对照，输出 CSV；
- 在不改变类别顺序的约束下比较两种校准集规模。

