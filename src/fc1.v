///////////////////////////////////////////////////////////////////////////////
// 模块: fc1
// 功能: 全连接层第一层，完成矩阵-向量乘法 Y = W * X
//       输入: 484 个 int8 像素 (0~127)
//       权重: 118x484，每个权重为 int4 (-7~7)
//       输出: 118 个 int8 (0~127)，经过 ReLU 和定点量化
// 量化公式: output = clamp( (relu(acc) * 1557 + 32768) >> 16 , 0, 127 )
///////////////////////////////////////////////////////////////////////////////

module fc1 (
    // 时钟与复位
    input  wire         clk,
    input  wire         rst,
    input  wire         en,                     // 模块使能

    // AXI Stream 输入接口 (来自上一层或输入缓冲)
    input  wire signed [7:0]   data_in_tdata,   // 输入数据 (int8, 非负)
    input  wire         data_in_tvalid,         // 输入有效
    output wire         data_in_tready,         // 模块可以接收数据

    // 权重存储器接口 (组合读，无延迟)
    output reg  [8:0]   weight_addr,            // 读地址 (0~483)
    input  wire [471:0] weight_data_col,        // 读出的权重列 (118×4bit)

    // AXI Stream 输出接口 (发往 fc2 或激活)
    output wire signed [7:0]   data_out_tdata,  // 输出数据 (int8, 0~127)
    output wire         data_out_tvalid,        // 输出有效
    input  wire         data_out_tready         // 下游准备好接收
);

    //====================== 状态机定义 ======================
    localparam IDLE       = 2'b00;   // 空闲
    localparam CALC       = 2'b01;   // 计算 (接收输入 + 乘累加)
    localparam RELU       = 2'b10;   // ReLU + 量化
    localparam WRITE_BACK = 2'b11;   // 串行输出结果

    reg [1:0] state, next_state;

    //====================== 计数器 ======================
    reg [8:0] in_cnt;       // 已接收输入个数 (0~484)
    reg [6:0] out_cnt;      // 已输出结果个数 (0~117)
    reg [8:0] col_cnt;      // 乘累加阶段处理的列数 (0~484)

    //====================== 存储与累加 ======================
    reg signed [7:0] input_buf [0:483];   // 存储 484 个输入像素
    reg signed [31:0] acc [0:117];        // 118 个神经元的累加器 (int32)
    reg [7:0] result [0:117];             // 量化后的最终结果 (uint8)

    //====================== 权重拆分 ======================
    // weight_data_col 为 472-bit，每 4-bit 为一个有符号权重 (int4)
    // weight_col[i] 对应第 i 个神经元的权重 (i=0~117)
    wire signed [3:0] weight_col [0:117];
    genvar i;
    generate
        for (i = 0; i < 118; i = i + 1) begin
            assign weight_col[i] = $signed(weight_data_col[i*4+3 : i*4]);
        end
    endgenerate

    //====================== 输入流控 ======================
    // data_in_ready 为高时才能接收新数据
    // 空闲状态或计算状态且尚未收满 484 个输入时，允许接收
    assign data_in_tready = (state == IDLE) || (state == CALC && in_cnt < 9'd484);

    //====================== 状态机时序逻辑 ======================
    always @(posedge clk or posedge rst) begin
        if (rst) state <= IDLE;
        else state <= next_state;
    end

    //====================== 状态机组合逻辑（状态跳转） ======================
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:
                // 使能有效时立即进入 CALC 状态 (等待输入)
                if (en) next_state = CALC;

            CALC:
                // 输入已收满 (484 个) 且乘累加也已处理完 484 列时，进入 RELU
                if (in_cnt == 9'd484 && col_cnt == 9'd484)
                    next_state = RELU;

            RELU:
                // RELU+量化 只需一个周期，立即进入 WRITE_BACK
                next_state = WRITE_BACK;

            WRITE_BACK:
                // 所有 118 个结果发送完毕且下游握手成功，返回 IDLE
                if (out_cnt == 7'd117 && data_out_tready)
                    next_state = IDLE;
        endcase
    end

    //====================== 输入接收阶段 ======================
    // 将输入数据存入 input_buf，同时计数 in_cnt
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            in_cnt <= 9'd0;
        end else if (state == IDLE) begin
            in_cnt <= 9'd0;
        end else if (state == CALC && data_in_tvalid && in_cnt < 9'd484) begin
            input_buf[in_cnt] <= data_in_tdata;
            in_cnt <= in_cnt + 9'd1;
        end
    end

    //====================== 乘累加阶段：列地址生成 ======================
    // 当所有输入接收完毕后 (in_cnt==484)，开始逐列输出 weight_addr
    // 每个周期输出一列地址，col_cnt 从 0 递增到 484
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            col_cnt <= 9'd0;
            weight_addr <= 9'd0;
        end else if (state == IDLE) begin
            col_cnt <= 9'd0;
            weight_addr <= 9'd0;
        end else if (state == CALC && in_cnt == 9'd484 && col_cnt < 9'd484) begin
            weight_addr <= col_cnt;          // 输出当前列地址
            col_cnt <= col_cnt + 9'd1;       // 列计数加一
        end
    end

    //====================== 乘累加计算 ======================
    // 每个有效周期，对 118 个神经元并行累加
    // 注意：当前周期的 weight_col 对应地址 col_cnt-1 的权重
    integer j, col;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 118; j = j + 1) acc[j] <= 32'd0;
        end else if (state == IDLE) begin
            for (j = 0; j < 118; j = j + 1) acc[j] <= 32'd0;
        end else if (state == CALC && in_cnt == 9'd484 && col_cnt > 0) begin
            col = col_cnt - 1;   // 当前处理的列索引
            for (j = 0; j < 118; j = j + 1)
                acc[j] <= acc[j] + (weight_col[j] * input_buf[col]);
        end
    end

    //====================== ReLU + 定点量化 ======================
    // 严格按照 plan.md 要求：
    //   output = clamp( (relu(acc) * 1557 + 32768) >> 16 , 0, 127 )
    // 实现步骤:
    //   1. 负数归零 (ReLU)
    //   2. 乘以 1557 (扩展至 64 位防溢出)
    //   3. 加 32768 (舍入常数)
    //   4. 右移 16 位
    //   5. 饱和到 [0,127]
    // 所有操作在一个周期内完成 (118 个神经元并行)
    localparam signed [31:0] SCALE_MUL = 32'd1557;   // 乘数 M
    localparam integer       SHIFT     = 16;         // 右移位数
    localparam signed [31:0] ROUND     = 32'd32768;  // 舍入常数 = 2^(SHIFT-1)

    reg signed [63:0] prod;      // 乘积 (64-bit)
    reg signed [31:0] scaled;    // 缩放后结果 (32-bit)
    reg [7:0] tmp;               // 临时结果

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 118; j = j + 1) result[j] <= 8'd0;
        end else if (state == RELU) begin
            for (j = 0; j < 118; j = j + 1) begin
                // ----- ReLU: 负数归零 -----
                if (acc[j] < 32'd0) begin
                    tmp = 8'd0;
                end else begin
                    // ----- 定点缩放 -----
                    // 将 32-bit 无符号扩展为 64-bit 以避免乘法溢出
                    prod = {32'd0, acc[j]} * SCALE_MUL;
                    // 加舍入常数 (实现 round to nearest)
                    prod = prod + ROUND;
                    // 右移 SHIFT 位 (无符号右移，因为 prod 为正)
                    scaled = prod >>> SHIFT;
                    // ----- 饱和到 0~127 -----
                    if (scaled > 32'd127)
                        tmp = 8'd127;
                    else
                        tmp = scaled[7:0];   // 低 8 位即为结果
                end
                result[j] <= tmp;
            end
        end
    end

    //====================== 输出阶段 ======================
    // 将 result 数组中的 118 个结果串行发送出去
    // out_cnt 逐个递增，配合 data_out_tready 握手
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_cnt <= 7'd0;
        end else if (state == IDLE) begin
            out_cnt <= 7'd0;
        end else if (state == WRITE_BACK && data_out_tready && out_cnt < 7'd117) begin
            out_cnt <= out_cnt + 7'd1;
        end
    end

    // 输出数据: 根据 out_cnt 索引选择对应的结果
    // 由于 result 为无符号 0~127，这里转换为 signed 8-bit (实际值不变)
    assign data_out_tdata = $signed({1'b0, result[out_cnt]});
    assign data_out_tvalid = (state == WRITE_BACK);

endmodule
