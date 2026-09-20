# 最小案例 2：端侧视觉模型量化（sin 模型）

> 类型：`PY` ｜ 验证状态：待验证 ｜ 前置：[`microcourses/01_train_sin_model`](../01_train_sin_model/)

## 摘要

最小案例 1 训出的模型用的是 float32 浮点数：每个参数占 4 字节，ESP32-S3 这类单片机算浮点又慢又费电。本案例用乐鑫的量化工具 **ESP-PPQ** 把模型换成 8 位整数（INT8），并导出成 ESP-DL 推理库能直接加载的 **`.espdl`** 格式。量化好比把精装书重排成口袋书：内容基本不变，体积大幅下降，代价是精度略损——代价多大，本案例用 MSE 实测量化前后误差、对比文件字节数，用数字回答"量化值不值"。产出的 `sin_model.espdl` 会在最小案例 3 由 ESP-DL 加载、在真实芯片上推理。**本案例只在电脑上运行，不需要开发板。**

## Quick Start

**预备知识**

- **已完成最小案例 1**：`../01_train_sin_model/outputs/` 下存在 `sin_model.onnx` 和 `sin_model.pth`，没有就先回去跑；
- **已激活 `esp32s3` conda 环境**：最小案例 1 的 requirements.txt 已含 `esp-ppq==1.2.10`，本目录没有单独的依赖文件；
- **不需要 GPU**：脚本固定 `DEVICE = "cpu"`；float32、INT8、MSE 等关键词下面会用大白话解释。


以下步骤从仓库根目录 `esi-mvp-code/` 开始。

**第 1 步：激活 conda 环境**（成功后提示符出现 `(esp32s3)`）

```bash
conda activate esp32s3
```

**第 2 步：进入案例目录**（`ls` 应能看到 `quantize_onnx_model.py`）

```bash
cd microcourses/02_quantize_sin_model
```

**第 3 步：确认/安装依赖**（本目录没有 requirements.txt，复用最小案例 1 的，其中 `esp-ppq==1.2.10` 就是本集的量化工具；装过则很快结束）

```bash
python -m pip install -r ../01_train_sin_model/requirements.txt
```

**第 4 步：检查前置产物**（`test -f` 无输出且退出码 0 表示文件存在；不存在就回最小案例 1 重训，**不要用空文件或改名绕过**）

```bash
test -f ../01_train_sin_model/outputs/sin_model.onnx
test -f ../01_train_sin_model/outputs/sin_model.pth
```

**第 5 步：运行量化**（脚本没有命令行参数，配置都在代码里；ESP-PPQ 会打印大量日志属正常，CPU 上约几十秒到几分钟）

```bash
python quantize_onnx_model.py
```

**第 6 步：核对结果**（应看到 `sin_model.espdl`、`sin_model.json`、`sin_model.info`；把终端最后四行——两组 MSE、两组字节数——抄进实验记录，quant MSE 略大于 float MSE 是正常代价）

```bash
ls outputs
```

## 预期现象

成功时终端末尾输出**格式**如下（格式示例，具体数值以本机运行为准；本案例验证状态为"待验证"，不预设固定 MSE 数字）：

```text
...（ESP-PPQ 自身的量化/导出日志，含量化报告）...
float model MSE: 0.xxxxx
quant model MSE: 0.xxxxx
model size: .../01_train_sin_model/outputs/sin_model.onnx = NNNN bytes
model size: outputs/sin_model.espdl = NNNN bytes
```

- 两组 MSE 用**同一份数据、同一个公式**，可直接对比：`float` 来自 `.pth` 浮点模型，`quant` 来自 PPQ 图模拟的量化推理；退出码 0 且三个产物齐全即成功，实际数值记入实验报告；
- 失败时：找不到 ONNX/PTH 回最小案例 1 重新生成；导入或导出失败检查依赖版本、shape `[1, 1]` 和目标 `esp32s3`；误差异常确认 `shuffle=False` 并保留完整报告。

**常见问题排查**

| 现象 | 原因 | 解决办法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'esp_ppq'` | 没激活环境或没装依赖 | 激活后安装最小案例 1 的 requirements.txt，用 `python -c "import esp_ppq"` 验证 |
| `ModuleNotFoundError: No module named 'sin_model'` | 没在本目录运行，或最小案例 1 目录被移动/改名 | 保持两集目录名和相对位置不变 |
| `FileNotFoundError: .../sin_model.onnx`（或 .pth） | 最小案例 1 没运行或产物被删 | 回最小案例 1 重新训练，不要绕过 |
| 量化报 shape 不匹配 | `INPUT_SHAPE` 与 ONNX 实际输入不一致 | 保持 `[1, 1]`；改过模型结构要两边同步 |
| quant MSE 远大于 float MSE（差一个数量级以上） | 校准数据分布不对或被改成 `shuffle=True` | 确认校准数据仍由 `generate_data()` 生成且 `shuffle=False` |
| `.espdl` 生成了但最小案例 3 板上自检失败 | `export_test_values` 被关，或 ONNX 与 PTH 不是同一次训练的产物 | 保持 `True`；最小案例 1 重训后必须重新量化，不能混用新旧产物 |

## 学习目标

- 运行量化脚本前自查前置模型文件是否存在；
- 用自己的话解释什么是量化、什么是校准数据，以及校准数据为什么必须与模型输入分布匹配；
- 从终端读出 float/quant 两组 MSE 和两个文件的字节数，填入实验记录；
- 说出 `INPUT_SHAPE`、`TARGET`、`NUM_OF_BITS`、`calib_steps` 各控制什么；
- 判断量化产物是否满足最小案例 3 的部署要求。

## 原理讲解

**量化：把 float32 压成 INT8**

模型里的权重和中间结果大多是 float32，精细但占存储、算得慢、耗电多。量化就是把它们映射到 INT8（-128~127，共 256 个档位）。类比：float32 像精确到毫米的卷尺，INT8 像只有 256 个刻度的表盘——量程选得好，日常用途完全够，而且读数快、省电。收益：参数从每个 4 字节压到 1 字节左右，且 ESP32-S3 有专门的 INT8 指令，整数运算快得多；代价：映射必然有舍入误差，所以要实测精度掉了多少。

![浮点数到 INT8 的量化示意图（AI 生成示意）](../../docs/assets/ai_generated/float_to_int8_quantization.png)

> 图：量化通过校准输入范围，把连续的浮点表示映射到有限的整数表示。图中数字仅用于解释概念，实际参数以 ESP-PPQ 报告为准。

**校准：决定 256 个档位怎么画**

范围画小了，大数值被"削顶"；画大了，刻度太稀、小数值挤在一起。**校准（Calibration）**就是拿一批有代表性的输入喂给浮点模型，统计每层实际出现的数值范围，据此确定量化参数（scale/exponent，记录在 `sin_model.json` 里）。这叫"训练后量化"（PTQ），不需要重新训练。

校准数据必须和真实输入同分布。本脚本直接 `from sin_model import generate_data`，用最小案例 1**同一个函数**生成校准数据，从根上保证分布一致；换成随机乱数或别的范围，量化刻度就会画错。

**ESP-PPQ 与三个产物**

ESP-PPQ 是乐鑫定制的量化工具，一次调用 `espdl_quantize_onnx()` 就完成"读 ONNX → 校准 → INT8 量化 → 导出"。产物三个：

- `sin_model.espdl`：ESP-DL 可直接加载的量化模型（最小案例 3 的主角）；
- `sin_model.json`：量化参数（scale/exponent 等）；
- `sin_model.info`：人类可读的量化报告，输入层已标成 `INT8, 1x1`。

**两个容易被忽略的细节**

1. **`shuffle=False`**：计算量化误差会多次遍历数据集，若打乱顺序，每次看到的数据不同，得到的误差就是错的；
2. **`export_test_values=True`**：把一组测试输入/输出嵌进 `.espdl`，最小案例 3 板上 `model->test()` 加载自检依赖它。

## 整体流程图

```text
../01_train_sin_model/outputs/sin_model.onnx + sin_model.pth（同一次训练的产物）
             ↓ generate_data() 生成校准数据（shuffle=False）
             ↓ espdl_quantize_onnx(target=esp32s3, num_of_bits=8, calib_steps=32)
outputs/sin_model.espdl + sin_model.json + sin_model.info
             ↓ 对同一份数据分别评估 float MSE（.pth）/ quant MSE（PPQ 图模拟量化）
.onnx vs .espdl 文件字节数对照 → 写入实验记录
```

## 关键代码解析

片段来自 [`quantize_onnx_model.py`](quantize_onnx_model.py)。

**1. 复用最小案例 1 代码 + 只取 x 的 collate_fn**

```python
TRAIN_DIR = Path(__file__).resolve().parents[1] / "01_train_sin_model"
sys.path.insert(0, str(TRAIN_DIR))        # 把最小案例 1 目录加入模块搜索路径
from sin_model import SinPredictor, generate_data

def collate_fn(batch):
    # 数据集每次返回 (x, y)，校准只需要输入 x
    return torch.stack([sample[0] for sample in batch]).to(DEVICE)
```

校准数据与训练数据由同一个函数生成，分布天然一致；不改 `collate_fn` 的话，标签会被错误地当成模型输入喂给校准器。

**2. 量化主调用：关键配置集中在一处**

```python
quant_ppq_graph = espdl_quantize_onnx(
    onnx_import_file=ONNX_MODEL_PATH,     # 输入：最小案例 1 的 ONNX
    espdl_export_file=ESPDL_MODEL_PATH,   # 输出：outputs/sin_model.espdl
    calib_dataloader=dataloader,          # 校准数据（shuffle=False！）
    calib_steps=32,                       # 用 32 个 batch 做校准
    input_shape=INPUT_SHAPE,              # [1, 1]，batch 必须为 1
    target=TARGET,                        # "esp32s3"
    num_of_bits=NUM_OF_BITS,              # 8，即 INT8
    collate_fn=collate_fn,
    error_report=True,                    # 打印量化误差报告
    export_test_values=True,              # 供最小案例 3 板上 model->test() 自检
    verbose=1,
)
```

**3. 量化前后 MSE 对照**

```python
# 浮点模型（量化前）：加载 .pth 算 MSE
model.load_state_dict(torch.load(PTH_MODEL_PATH, weights_only=True))
print(f"float model MSE: {loss.item():.5f}")

# 量化模型（量化后）：TorchExecutor 模拟 INT8 推理
executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)
y_pred = executor(batch_x)
print(f"quant model MSE: {loss.item():.5f}")
```

`TorchExecutor` 的前向会模拟 INT8 舍入，两组数字放在一起，才能把量化引入的额外误差单独剥离出来。

## 关键文件说明

| 文件/目录 | 职责 | 类型 |
|---|---|---|
| [`quantize_onnx_model.py`](quantize_onnx_model.py) | 校准、量化、误差和大小对照（本集入口） | 课程代码 |
| [`../01_train_sin_model/sin_model.py`](../01_train_sin_model/sin_model.py) | 提供 `SinPredictor` 和 `generate_data` | 共享课程代码 |
| `../01_train_sin_model/outputs/sin_model.onnx` | 量化输入（ONNX 模型） | 上游生成物 |
| `../01_train_sin_model/outputs/sin_model.pth` | 浮点对照评估的权重 | 上游生成物 |
| `outputs/sin_model.espdl` | ESP-DL 部署模型（最小案例 3 输入） | 生成物 |
| `outputs/sin_model.json` | 量化参数（scale/exponent 等） | 生成物 |
| `outputs/sin_model.info` | 人类可读的量化图报告 | 生成物 |

## 配置说明

配置都直接写在 `quantize_onnx_model.py` 里，改后重跑即可：

| 配置项 | 现象变化 |
|---|---|
| `INPUT_SHAPE=[1,1]` | 必须与模型输入匹配，改错会量化失败或产物不可用 |
| `TARGET="esp32s3"` | 目标芯片，可选 `'c'`、`'esp32s3'`、`'esp32p4'` |
| `NUM_OF_BITS=8` | 量化位宽，影响模型大小和误差 |
| `calib_steps=32` | 校准用的 batch 数，增大覆盖更广但更耗时 |
| `DEVICE="cpu"` | 改 GPU 前先确认 CUDA 可用 |
| `export_test_values=True` | 关掉后最小案例 3 板上 `model->test()` 无法自检 |

## 术语小表

| 术语 | 解释 |
|---|---|
| 量化 | 把模型里的 float32 换成 INT8，省存储、算得快，代价是少量精度 |
| INT8 | 8 位有符号整数，只有 -128~127 共 256 个档位 |
| 校准 | 用代表性输入跑模型、统计各层数值范围，确定量化刻度（scale/exponent） |
| 校准数据 | 用来校准的那批输入，必须和真实输入同分布；本例复用最小案例 1 的 `generate_data()` |
| 训练后量化（PTQ） | 不重新训练、只用少量数据校准就完成量化 |
| ESP-PPQ | 乐鑫的训练后量化工具，输入 ONNX、输出 `.espdl` |
| `.espdl` / ESP-DL | ESP-DL 推理库可加载的量化模型文件格式 / 乐鑫的端侧推理库（最小案例 3 用） |
| calib_steps | 校准用多少个 batch；本例 32 |
| scale/exponent | 量化刻度参数，记录在 `sin_model.json`，端侧用它把 INT8 还原成近似浮点值 |
| shuffle=False | 不打乱数据顺序，保证多次遍历数据集时结果可比较 |

## 验证清单

- [ ] ONNX 和 PTH 均存在且来自同一次训练；
- [ ] 记录目标芯片（esp32s3）、shape（[1,1]）、校准步数（32）和位宽（8）；
- [ ] 记录 float/quant MSE 和 ONNX/ESPDL 字节数到实验表格；
- [ ] `outputs/` 中 `.espdl`、`.json`、`.info` 三个文件齐全；
- [ ] `sin_model.espdl` 可被最小案例 3 复制或嵌入；
- [ ] 不提交个人数据或未经许可的校准素材。

## 思考题与拓展挑战

**思考题**

1. 为什么校准数据不应随意打乱或换成完全不同的分布？

   **参考答案：**校准器要从代表性输入估计数值范围；分布不匹配会让量化刻度盖不住真实输入，推理误差变大。

2. 为什么同时比较 MSE 和模型大小？

   **参考答案：**只有误差看不出部署收益，只有体积看不出精度；两个指标一起权衡，才能说明量化方案是否适合端侧。

3. 为什么计算量化误差时 DataLoader 必须 `shuffle=False`？

   **参考答案：**浮点和量化评估会遍历同一个 dataloader，打乱后两次看到的数据不一致，MSE 差值里就混入了"数据不同"的因素，无法归因于量化本身。

**拓展挑战**

- 设计两组校准数据规模，在保持输出文件命名不变的约束下比较误差和耗时；
- 为量化报告增加一个自动化的"是否超过误差阈值"字段。

