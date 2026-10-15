# ctmodels 模型使用

六种模型均可通过 `build_model` 构建。在 `models` 目录运行下面的示例，或将该目录加入 Python 模块搜索路径。

## 公共参数与输入输出

构建接口：`build_model(model_name, input_size, hidden_size, output_size=None, **kwargs)`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `model_name` | 必填 | 模型名称：`lstm`、`ctrnn`、`ltc`、`cfc`、`ctgru`、`node`，不区分大小写 |
| `input_size` | 必填 | 每个时间点的输入特征维度 `F`，正整数 |
| `hidden_size` | 必填 | 隐藏单元数量 `H`，正整数 |
| `output_size` | `None` | 每个时间点的输出维度 `O`；`None` 时 `O=H`，指定正整数时增加 `H → O` 的线性输出层，状态维度不变 |
| `batch_first` | `True` | `True` 使用 `(B,T,F)` 输入；`False` 使用 `(T,B,F)` 输入 |
| `return_sequences` | `True` | `True` 返回每一步输出；`False` 只返回最后一步输出 |

`B` 是 batch size，`T` 是序列长度。输出层为线性层，输出值不自动转换为概率或限制到坐标范围。

调用接口：`output, state = model(x, hx=None, timespans=None)`。

| 参数 / 返回值 | 含义 |
| --- | --- |
| `x` | 浮点输入特征序列，最后一维必须等于 `input_size`；batch 和序列长度必须大于零 |
| `hx` | 初始状态，形状与该模型返回的 `state` 相同；`None` 时从全零状态开始。传入张量须与 `x` 的 dtype、device 一致 |
| `timespans` | 每一步之前经过的时间 `Δt`，为有限且严格大于零的值；传入的是时间间隔，而非绝对时间戳 |
| `output` | 序列输出或最后一步输出，形状见下表 |
| `state` | 最后一步的内部状态，可作为下一段序列的 `hx`；各模型的形状见对应使用说明 |

| 输入布局 | `x` | `return_sequences=True` 的输出 | `return_sequences=False` 的输出 |
| --- | --- | --- | --- |
| `batch_first=True` | `(B,T,F)` | `(B,T,O)` | `(B,O)` |
| `batch_first=False` | `(T,B,F)` | `(T,B,O)` | `(B,O)` |
| 无 batch，任意 `batch_first` | `(T,F)` | `(T,O)` | `(O,)` |

`timespans` 支持以下形式，会自动转换到 `x` 的 dtype、device：

| 形式 | 含义 |
| --- | --- |
| 标量，例如 `0.1` | 所有样本、所有步骤使用同一个间隔 |
| `(T,)` | 所有样本共用一组时间间隔 |
| `(B,T)` 或 `(B,T,1)` | `batch_first=True` 时，每个样本使用自己的时间间隔 |
| `(T,B)` 或 `(T,B,1)` | `batch_first=False` 时，每个样本使用自己的时间间隔 |
| `(T,)` 或 `(T,1)` | 无 batch 输入的时间间隔 |

下面每种模型的示例均使用 `F=128`、`H=64`、`O=3`、`B=4`、`T=20`。

## LSTM

### 构建与调用

```python
import torch
from ctmodels import build_model

model = build_model(
    "lstm",
    input_size=128,
    hidden_size=64,
    output_size=3,
    batch_first=True,
    return_sequences=True,
)

x = torch.randn(4, 20, 128)
output, (h, c) = model(x)

print(output.shape)  # (4, 20, 3)
print(h.shape)       # (4, 64)
print(c.shape)       # (4, 64)
```

也可直接构建：`from ctmodels import LSTM`，然后调用 `LSTM(128, 64, output_size=3)`。

### 参数与输入输出

构造参数为公共参数表中的 `input_size`、`hidden_size`、`output_size`、`batch_first`、`return_sequences`。
LSTM 接收 `timespans`，但门控计算不使用时间间隔。

| 项目 | 形状 / 含义 |
| --- | --- |
| 输入 `x` | 默认 `(B,T,F)` |
| 输出 `output` | 默认 `(B,T,O)`；`return_sequences=False` 时 `(B,O)` |
| 初始状态 `hx` | `(h0,c0)`，各为 `(B,H)`；省略时均为零 |
| 最终状态 `state` | `(h,c)`，`h` 是隐藏状态，`c` 是记忆状态，各为 `(B,H)` |
| 无 batch 的状态 | `(h,c)`，各为 `(H,)` |

## CTRNN

### 构建与调用

```python
import torch
from ctmodels import build_model

model = build_model(
    "ctrnn",
    input_size=128,
    hidden_size=64,
    output_size=3,
    batch_first=True,
    return_sequences=True,
    global_feedback=False,
    fix_tau=True,
    tau=1.0,
    ode_unfolds=6,
    delta_t=0.1,
    cell_clip=-1,
)

x = torch.randn(4, 20, 128)
dt = torch.full((4, 20), 0.1)
output, state = model(x, timespans=dt)

print(output.shape)  # (4, 20, 3)
print(state.shape)   # (4, 64)
```

也可直接构建：`from ctmodels import CTRNN`，然后调用 `CTRNN(128, 64, output_size=3)`。

### 专用参数

公共构造参数见开头的参数表。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `global_feedback` | `False` | 是否将当前隐藏状态拼入驱动网络输入；`False` 时驱动只取输入特征，`True` 时驱动同时取输入特征和隐藏状态 |
| `fix_tau` | `True` | 是否固定时间常数；`False` 时学习一个经 softplus 转换为正值的时间常数 |
| `tau` | `1.0` | 正时间常数；`fix_tau=True` 时直接使用该值，`False` 时它是可训练原始参数的初值，有效初始时间常数为 `softplus(tau)` |
| `ode_unfolds` | `6` | 每个时间点的 Euler 求解子步数，正整数 |
| `delta_t` | `0.1` | 未传 `timespans` 时的子步时长；默认每个时间点经过 `ode_unfolds * delta_t` 的时间 |
| `cell_clip` | `-1` | 大于零时，每个子步将内部状态裁剪到 `[-cell_clip, cell_clip]`；小于等于零时关闭裁剪 |

显式传入 `timespans` 后，求解子步时长为 `Δt / ode_unfolds`。不传时，默认每一步时长为 `0.6`。

### 输入输出

| 项目 | 形状 |
| --- | --- |
| 输入 `x` | 默认 `(B,T,F)` |
| 输出 `output` | 默认 `(B,T,O)`；只返回最后一步时 `(B,O)` |
| 初始状态 `hx` / 最终状态 `state` | `(B,H)`；无 batch 时 `(H,)` |

## LTC

### 构建与调用

```python
import torch
from ctmodels import build_model

model = build_model(
    "ltc",
    input_size=128,
    hidden_size=64,
    output_size=3,
    batch_first=True,
    return_sequences=True,
    mixed_memory=False,
    input_mapping="affine",
    output_mapping="affine",
    ode_unfolds=6,
    epsilon=1e-8,
    implicit_param_constraints=True,
    erev_init_seed=1111,
)

x = torch.randn(4, 20, 128)
dt = torch.full((4, 20), 0.1)
output, state = model(x, timespans=dt)

print(output.shape)  # (4, 20, 3)
print(state.shape)   # (4, 64)
```

也可直接构建：`from ctmodels import LTC`，然后调用 `LTC(128, 64, output_size=3)`。

### 专用参数

公共构造参数见开头的参数表。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `mixed_memory` | `False` | 是否在 LTC 更新前加入 LSTM 记忆更新；开启后 `hx` 和 `state` 都使用 `(h,c)` 元组 |
| `input_mapping` | `"affine"` | 输入的逐特征映射：`"affine"` 为可训练缩放加偏置，`"linear"` 为可训练缩放，`None` / `"none"` 为直接使用输入 |
| `output_mapping` | `"affine"` | LTC 隐藏输出的逐特征映射，选项与 `input_mapping` 相同；该映射位于公共 `output_size` 线性输出层之前 |
| `ode_unfolds` | `6` | 每个时间点的半隐式求解子步数，正整数；子步时长为 `Δt / ode_unfolds` |
| `epsilon` | `1e-8` | ODE 更新分母中的正数稳定项，用于避免除零 |
| `implicit_param_constraints` | `True` | `True` 时在计算中用 softplus 保证电容、电导等参数为正；`False` 时需在每次 `optimizer.step()` 后调用 `model.apply_weight_constraints()` |
| `erev_init_seed` | `1111` | 反转电位参数初始正负号的随机种子，仅控制这部分初始化 |

不传 `timespans` 时，每一步时长为 `1.0`。

### 输入输出

| 项目 | 形状 / 含义 |
| --- | --- |
| 输入 `x` | 默认 `(B,T,F)` |
| 输出 `output` | 默认 `(B,T,O)`；只返回最后一步时 `(B,O)` |
| 默认初始状态 `hx` / 最终状态 `state` | `(B,H)`；无 batch 时 `(H,)` |
| `mixed_memory=True` 的状态 | `(h,c)`，各为 `(B,H)`；无 batch 时各为 `(H,)` |

## CfC

### 构建与调用

```python
import torch
from ctmodels import build_model

model = build_model(
    "cfc",
    input_size=128,
    hidden_size=64,
    output_size=3,
    batch_first=True,
    return_sequences=True,
    mixed_memory=False,
    mode="default",
    activation="lecun_tanh",
    backbone_units=128,
    backbone_layers=1,
    backbone_dropout=0.0,
)

x = torch.randn(4, 20, 128)
dt = torch.full((4, 20), 0.1)
output, state = model(x, timespans=dt)

print(output.shape)  # (4, 20, 3)
print(state.shape)   # (4, 64)
```

也可直接构建：`from ctmodels import CfC`，然后调用 `CfC(128, 64, output_size=3)`；`CFC` 是同一类的别名。

### 专用参数

公共构造参数见开头的参数表。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `mixed_memory` | `False` | 是否在 CfC 更新前加入 LSTM 记忆更新；开启后 `hx` 和 `state` 都使用 `(h,c)` 元组 |
| `mode` | `"default"` | 更新方式：`"default"` 用时间门控混合两个候选状态；`"pure"` 使用单分支指数闭式更新；`"no_gate"` 将第一候选状态与时间门控后的第二候选状态相加 |
| `activation` | `"lecun_tanh"` | backbone 激活函数，可选 `"lecun_tanh"`、`"silu"`、`"relu"`、`"tanh"`、`"gelu"` |
| `backbone_units` | `128` | backbone 每层的特征维度；启用 backbone 时必须为正整数 |
| `backbone_layers` | `1` | backbone 的层数，非负整数；`0` 表示关闭 backbone，直接使用输入与隐藏状态的拼接特征 |
| `backbone_dropout` | `0.0` | backbone 中第 2 层及后续各层激活后的 dropout 概率，范围 `[0,1)`；层数小于等于 1 时不生效 |

不传 `timespans` 时，每一步时长为 `1.0`。

### 输入输出

| 项目 | 形状 / 含义 |
| --- | --- |
| 输入 `x` | 默认 `(B,T,F)` |
| 输出 `output` | 默认 `(B,T,O)`；只返回最后一步时 `(B,O)` |
| 默认初始状态 `hx` / 最终状态 `state` | `(B,H)`；无 batch 时 `(H,)` |
| `mixed_memory=True` 的状态 | `(h,c)`，各为 `(B,H)`；无 batch 时各为 `(H,)` |

## CTGRU

### 构建与调用

```python
import torch
from ctmodels import build_model

model = build_model(
    "ctgru",
    input_size=128,
    hidden_size=64,
    output_size=3,
    batch_first=True,
    return_sequences=True,
    M=8,
    cell_clip=-1,
)

x = torch.randn(4, 20, 128)
dt = torch.full((4, 20), 0.1)
output, state = model(x, timespans=dt)

print(output.shape)  # (4, 20, 3)
print(state.shape)   # (4, 512)，64 * 8
```

也可直接构建：`from ctmodels import CTGRU`，然后调用 `CTGRU(128, 64, output_size=3)`。

### 专用参数

公共构造参数见开头的参数表。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `M` | `8` | 每个隐藏单元保存的时间尺度数量，正整数；时间常数按 `10**(m/2)` 排列，`m=0,...,M-1`；内部状态维度为 `H*M` |
| `cell_clip` | `-1` | 大于零时，将每个时间尺度的内部状态裁剪到 `[-cell_clip, cell_clip]`；小于等于零时关闭裁剪 |

不同时间尺度的内部状态求和后得到 `H` 维隐藏输出，再经可选的线性输出层得到 `O` 维输出。
不传 `timespans` 时，每一步时长为 `1.0`。

### 输入输出

| 项目 | 形状 |
| --- | --- |
| 输入 `x` | 默认 `(B,T,F)` |
| 输出 `output` | 默认 `(B,T,O)`；只返回最后一步时 `(B,O)` |
| 初始状态 `hx` / 最终状态 `state` | `(B,H*M)`；无 batch 时 `(H*M,)` |

## NODE

### 构建与调用

```python
import torch
from ctmodels import build_model

model = build_model(
    "node",
    input_size=128,
    hidden_size=64,
    output_size=3,
    batch_first=True,
    return_sequences=True,
    ode_unfolds=6,
    delta_t=0.1,
    cell_clip=-1,
)

x = torch.randn(4, 20, 128)
dt = torch.full((4, 20), 0.1)
output, state = model(x, timespans=dt)

print(output.shape)  # (4, 20, 3)
print(state.shape)   # (4, 64)
```

也可直接构建：`from ctmodels import NODE`，然后调用 `NODE(128, 64, output_size=3)`。

### 专用参数

公共构造参数见开头的参数表。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `ode_unfolds` | `6` | 每个时间点的 RK4 求解子步数，正整数 |
| `delta_t` | `0.1` | 未传 `timespans` 时的子步时长；默认每个时间点经过 `ode_unfolds * delta_t` 的时间 |
| `cell_clip` | `-1` | 大于零时，每个子步将内部状态裁剪到 `[-cell_clip, cell_clip]`；小于等于零时关闭裁剪 |

显式传入 `timespans` 后，求解子步时长为 `Δt / ode_unfolds`。不传时，默认每一步时长为 `0.6`。

### 输入输出

| 项目 | 形状 |
| --- | --- |
| 输入 `x` | 默认 `(B,T,F)` |
| 输出 `output` | 默认 `(B,T,O)`；只返回最后一步时 `(B,O)` |
| 初始状态 `hx` / 最终状态 `state` | `(B,H)`；无 batch 时 `(H,)` |
