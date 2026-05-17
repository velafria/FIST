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
