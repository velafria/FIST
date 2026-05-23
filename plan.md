在 FPGA 设计中，状态机（FSM）是模块的“大脑”。针对你定义的四个阶段，我们需要精确描述每一个时钟周期内硬件的行为、计数器的变化以及信号的握手逻辑。
以下是各个阶段的详细语义定义：
1. IDLE (空闲阶段)
• 语义： 模块处于复位或等待状态，不消耗多余功耗，准备好接收新任务。
• 核心动作： • 重置计数器： 将输入像素计数器（in_cnt）和输出神经元计数器（out_cnt）清零。 • 清除累加器： 将 118 个 acc 寄存器置 0。 • 信号输出： 拉高 data_in_ready，表示可以接收数据；拉低 data_out_valid 和 done。
• 跳转逻辑： 当 en == 1 且 data_in_valid == 1 时，跳转至 CALC。
2. CALC (累加计算阶段)
• 语义： 执行 484 次迭代，完成矩阵-向量乘法 Y = W \cdot X。
• 核心动作： • 地址生成： 每个周期输出 weight_addr（指向第 in_cnt 列）。 • 流水线对齐： * T_0：发出地址。 • T_1：Regfile 返回该列 118 个权重，同时将 T_0 时刻输入的 data_in 延迟一拍送入乘法器。 • T_2：118 个乘累加器同时更新：acc[j] <= acc[j] + (data_in_reg * weight[j])。 • 计数维护： 当 data_in_valid 为高时，in_cnt 加 1。
• 跳转逻辑： 当 in_cnt == 483 且最后一个像素的累加完成后，跳转至 RELU。
3. RELU (激活与量化阶段)
• 语义： 对 118 个高位宽（如 20-bit）的累加结果进行非线性处理和位宽裁剪。
• 核心动作： • 偏置加法（可选）： 如果有 Bias，在此阶段将 118 个 Bias 加上。 • 激活逻辑： 对每个 acc[j] 进行判断：if (acc[j][MSB] == 1) acc[j] <= 0; (即负数归零)。 • 量化截断： 将 20-bit 的结果通过饱和截断（Saturate）或移位（Shift）转回 8-bit Int8。
• 说明： 如果硬件资源充足，这 118 个神经元的操作可以在 1 个时钟周期内并行完成。
• 跳转逻辑： 处理完成后，无条件跳转至 WRITE_BACK。
4. WRITE_BACK (结果输出阶段)
• 语义： 将存储在内部寄存器中的 118 个 8-bit 结果，通过窄总线（8-bit）依次发送给下游模块。
• 核心动作： • 并行转串行： 根据 out_cnt 指向的索引，从 118 个寄存器中选择一个送上 data_out 总线。 • 握手逻辑： * 拉高 data_out_valid。 • 如果下游 data_out_ready == 1，则 out_cnt 加 1。 • 如果下游不准备好，状态机在此“挂起”等待。
• 跳转逻辑： 当 out_cnt == 117 且最后一个数据成功发送后，跳转回 IDLE，或者拉高 done 信号。
状态转移总结表
当前状态	跳转条件	目标状态	关键输出信号
IDLE	en & data_in_valid	CALC	data_in_ready = 1
CALC	in_cnt == 483	RELU	weight_addr = in_cnt
RELU	计算完成 (通常 1 周期)	WRITE_BACK	acc_reg 更新为 8-bit
WRITE_BACK	out_cnt == 117 & ready	IDLE	data_out_valid = 1
设计建议：
1. 关于 RELU 阶段： 如果你追求极致性能，可以将 RELU 和量化逻辑直接合并到 CALC 的最后一个周期或者 WRITE_BACK 的输出逻辑中，从而省掉这个独立状态。但为了逻辑清晰，初版代码保留 RELU 状态更易于调试。
2. 关于计算： 由于输入是 484 个数，计算需要 484 个有效周期。请确保你的计数器逻辑能够正确处理 data_in_valid 抖动的情况（即输入不连续时，计算要暂停）。

这个结果表示你硬件里要用下面这套定点语义。

**1. 输入量化**

每张 22x22 图片单独归一化到 `0..127`：

```python
x_q = round(x_float / max(abs(x_float)) * 127)
x_q = clip(x_q, 0, 127)
```

硬件输入存储：

```text
input: 8bit unsigned 或 signed int8 非负
范围: 0..127
```

如果你从原始灰度 `0..255` 来，也按每张图最大值归一化到 127。

**2. FC1 权重量化**

`fc1.pth` 已经是 int4 量化后的权重：

```text
shape: [118, 484]
范围: -7..7
```

如果你需要从 float 权重重新量化：

```python
w1_q = round(w1_float / 0.11882889270782471)
w1_q = clip(w1_q, -7, 7)
```

硬件里直接用 `w1_q` 参与计算：

```text
fc1_acc[j] = sum_i input_q[i] * w1_q[j][i]
```

累加器建议 `int32`。

**3. FC1 输出量化**

这个最关键。`fc1_output_scale = 0.03936728154580424` 是浮点模型里的激活 scale。

如果硬件只用整数计算，那么 FC1 的整数累加结果 `fc1_acc_int` 和浮点激活之间关系近似是：

```text
fc1_float = fc1_acc_int * input_scale * fc1_weight_scale
```

其中每张图：

```text
input_scale = image_max_float / 127
```

如果你的 `x_float` 已经是 `0..1` 的图像，通常 `image_max_float` 接近 1，但严格来说是每张图自己的最大像素值。

FC1 输出量化目标是：

```text
fc1_out_q = round(relu(fc1_float) / fc1_output_scale)
fc1_out_q = clip(fc1_out_q, 0, 127)
```

代入整数累加：

```text
fc1_out_q = round(relu(fc1_acc_int) * input_scale * fc1_weight_scale / fc1_output_scale)
```

也就是：

```text
fc1_out_q = round(relu(fc1_acc_int) * K1)
```

其中：

```text
K1 = input_scale * 0.11882889270782471 / 0.03936728154580424
```

如果你输入就是按 `0..1` 且每张最大值约为 1 量化到 `0..127`，可以先用：

```text
input_scale = 1 / 127
K1 = (1 / 127) * 0.11882889270782471 / 0.03936728154580424
   ≈ 0.023755
```

所以硬件可以先近似：

```text
fc1_out_q = clamp(round(relu(fc1_acc_int) * 0.023755), 0, 127)
```

实现成乘法右移：

```text
fc1_out_q = clamp((relu_acc * M + round) >> SHIFT, 0, 127)
```

例如取：

```text
SHIFT = 16
M = round(0.023755 * 2^16) = 1557
```

即：

```verilog
fc1_out = saturate_to_127((relu_acc * 1557 + 32768) >> 16);
```

**4. FC2 权重量化**

`fc2.pth` 已经是：

```text
shape: [10, 118]
范围: -7..7
```

如果从 float 重新量化：

```python
w2_q = round(w2_float / 0.15770387649536133)
w2_q = clip(w2_q, -7, 7)
```

硬件 FC2：

```text
fc2_acc[k] = sum_j fc1_out_q[j] * w2_q[k][j]
```

然后直接：

```text
prediction = argmax(fc2_acc)
```

不用再反量化，因为 argmax 不受统一正比例缩放影响。

总结一下，你硬件主路径是：

```text
input_q[484]       : 0..127
fc1_w_q[118][484] : -7..7
fc1_acc int32
relu
fc1_out_q          : clamp(round(fc1_acc * K1), 0, 127)
fc2_w_q[10][118]  : -7..7
fc2_acc int32
argmax
```

其中当前可先用：

```text
K1 ≈ 0.023755
M = 1557
SHIFT = 16
```
## 硬件如何处理fc1 RELU后量化到int8
```
你需要的是：

```text
if scaled > 127:
    fc1_out = 127
else:
    fc1_out = scaled[7:0]
```

另外 rounding 常数建议用 `32768`，不是 `32767`。差别很小，但标准 round-to-nearest 对应：

```text
+ 2^(SHIFT-1)
```

也就是：

```text
+ 32768
```

**Verilog 写法**

假设：

```verilog
relu_acc : signed/unsigned 32bit，已经 ReLU 后非负
fc1_out  : 8bit，范围 0..127
```

可以这样写：

```verilog
localparam integer FC1_SCALE_MUL   = 1557;
localparam integer FC1_SCALE_SHIFT = 16;
localparam integer FC1_ROUND       = 1 << (FC1_SCALE_SHIFT - 1);

wire [31:0] relu_acc_u;
wire [63:0] scaled_mul;
wire [63:0] scaled_round;
wire [31:0] scaled_shift;

assign relu_acc_u     = acc[31] ? 32'd0 : acc;  // ReLU
assign scaled_mul     = relu_acc_u * FC1_SCALE_MUL;
assign scaled_round   = scaled_mul + FC1_ROUND;
assign scaled_shift   = scaled_round >> FC1_SCALE_SHIFT;

always @(*) begin
    if (scaled_shift > 32'd127)
        fc1_out = 8'd127;
    else
        fc1_out = scaled_shift[7:0];
end
```

如果你已经有单独的 ReLU：

```verilog
assign relu_acc_u = acc_relu;
```

那就不用 `acc[31] ? 0 : acc`。

**时序版推荐**

综合到 FPGA 时，我更建议打一两拍寄存器：

```verilog
localparam [15:0] FC1_SCALE_MUL = 16'd1557;
localparam integer FC1_SCALE_SHIFT = 16;

reg [63:0] scale_prod;
reg [31:0] scale_q;
reg [7:0]  fc1_out;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        scale_prod <= 64'd0;
        scale_q    <= 32'd0;
        fc1_out    <= 8'd0;
    end else begin
        // cycle 1: multiply + rounding
        scale_prod <= relu_acc_u * FC1_SCALE_MUL + 64'd32768;

        // cycle 2: shift
        scale_q <= scale_prod >> FC1_SCALE_SHIFT;

        // cycle 3: saturate
        if (scale_q > 32'd127)
            fc1_out <= 8'd127;
        else
            fc1_out <= scale_q[7:0];
    end
end
```

这段有 3 拍延迟，但时序会比纯组合更舒服。

**如果 relu_acc 是 signed**

先转非负 unsigned：

```verilog
wire [31:0] relu_acc_u;

assign relu_acc_u = acc_signed[31] ? 32'd0 : acc_signed[31:0];
```

然后后面全用 unsigned 做缩放。

结论：  
硬件 FC1 输出量化是：

```text
scaled = (relu_acc * 1557 + 32768) >> 16
fc1_out = min(scaled, 127)
```

不是取低八位，而是 **饱和到 127**。