///////////////////////////////////////////////////////////////////////////////
// 模块: fc2
// 功能: 全连接层第二层，完成矩阵-向量乘法并输出 argmax
//       输入: 118 个 int8 (来自 fc1 的输出，0~127)
//       权重: 10x118，每个权重为 int4 (-7~7)
//       输出: 8-bit 无符号整数，表示 argmax 索引 (0~9)
// 说明: 无需量化，argmax 不受缩放影响
///////////////////////////////////////////////////////////////////////////////

module fc2 (
    input  wire         clk,
    input  wire         rst,
    input  wire         en,

    // AXI Stream 输入接口 (来自 fc1)
    input  wire signed [7:0]   data_in_tdata,
    input  wire         data_in_tvalid,
    output wire         data_in_tready,

    // 权重存储器接口 (组合读，无延迟)
    output reg  [6:0]   weight_addr,       // 地址 0~117
    input  wire [39:0]  weight_data_col,   // 权重列: 10×4bit

    // AXI Stream 输出接口 (argmax 结果)
    output wire [7:0]   data_out_tdata,
    output wire         data_out_tvalid,
    input  wire         data_out_tready
);

    //====================== 状态机定义 ======================
    localparam IDLE       = 2'b00;   // 空闲
    localparam CALC       = 2'b01;   // 计算 (接收输入 + 乘累加)
    localparam ARGMAX     = 2'b10;   // 计算最大值索引
    localparam WRITE_BACK = 2'b11;   // 输出结果

    reg [1:0] state, next_state;

    //====================== 计数器 ======================
    reg [6:0] in_cnt;       // 已接收输入个数 (0~118)
    reg [6:0] col_cnt;      // 乘累加阶段处理的列数 (0~118)
    reg       out_done;     // 输出完成标志

    //====================== 存储与累加 ======================
    reg signed [7:0] input_buf [0:117];   // 存储 118 个输入 (fc1 输出)
    reg signed [31:0] acc [0:9];          // 10 个神经元的累加器 (int32)
    reg [7:0] result_idx;                 // argmax 结果 (0~9)

    //====================== 权重拆分 ======================
    // weight_data_col 为 40-bit，每 4-bit 为一个有符号权重 (int4)
    // weight_col[i] 对应第 i 个神经元的权重 (i=0~9)
    wire signed [3:0] weight_col [0:9];
    genvar i;
    generate
        for (i = 0; i < 10; i = i + 1) begin
            assign weight_col[i] = $signed(weight_data_col[i*4+3 : i*4]);
        end
    endgenerate

    //====================== 输入流控 ======================
    // data_in_tready 为高时才能接收新数据
    // 空闲状态或计算状态且尚未收满 118 个输入时，允许接收
    assign data_in_tready = (state == IDLE) || (state == CALC && in_cnt < 7'd118);

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
                // 使能有效时立即进入 CALC 状态
                if (en) next_state = CALC;

            CALC:
                // 输入已收满 (118 个) 且乘累加也已处理完 118 列时，进入 ARGMAX
                if (in_cnt == 7'd118 && col_cnt == 7'd118)
                    next_state = ARGMAX;

            ARGMAX:
                // argmax 计算只需一个周期，立即进入 WRITE_BACK
                next_state = WRITE_BACK;

            WRITE_BACK:
                // 输出完成且下游握手成功，返回 IDLE
                if (out_done && data_out_tready)
                    next_state = IDLE;
        endcase
    end

    //====================== 输入接收阶段 ======================
    // 将输入数据存入 input_buf，同时计数 in_cnt
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            in_cnt <= 7'd0;
        end else if (state == IDLE) begin
            in_cnt <= 7'd0;
        end else if (state == CALC && data_in_tvalid && in_cnt < 7'd118) begin
            input_buf[in_cnt] <= data_in_tdata;
            in_cnt <= in_cnt + 7'd1;
        end
    end

    //====================== 乘累加阶段：列地址生成 ======================
    // 当所有输入接收完毕后 (in_cnt==118)，开始逐列输出 weight_addr
    // 每个周期输出一列地址，col_cnt 从 0 递增到 118
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            col_cnt <= 7'd0;
            weight_addr <= 7'd0;
        end else if (state == IDLE) begin
            col_cnt <= 7'd0;
            weight_addr <= 7'd0;
        end else if (state == CALC && in_cnt == 7'd118 && col_cnt < 7'd118) begin
            weight_addr <= col_cnt;          // 输出当前列地址
            col_cnt <= col_cnt + 7'd1;       // 列计数加一
        end
    end

    //====================== 乘累加计算 ======================
    // 每个有效周期，对 10 个神经元并行累加
    // 当前周期的 weight_col 对应地址 col_cnt-1 的权重
    integer j, col;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 10; j = j + 1) acc[j] <= 32'd0;
        end else if (state == IDLE) begin
            for (j = 0; j < 10; j = j + 1) acc[j] <= 32'd0;
        end else if (state == CALC && in_cnt == 7'd118 && col_cnt > 0) begin
            col = col_cnt - 1;   // 当前处理的列索引
            for (j = 0; j < 10; j = j + 1)
                acc[j] <= acc[j] + (weight_col[j] * input_buf[col]);
        end
    end

    //====================== argmax 计算 ======================
    // 在 ARGMAX 状态的同一个周期内，使用组合逻辑找出累加值最大的神经元索引
    // 然后寄存到 result_idx 中
    reg [7:0] argmax_idx;
    reg [31:0] max_val;
    integer k;
    always @(*) begin
        argmax_idx = 0;
        max_val = acc[0];
        // 遍历所有 10 个神经元，找出最大值及其索引
        for (k = 1; k < 10; k = k + 1) begin
            if (acc[k] > max_val) begin
                max_val = acc[k];
                argmax_idx = k;
            end
        end
    end

    // 在 ARGMAX 状态将组合逻辑的结果寄存到输出寄存器
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result_idx <= 8'd0;
        end else if (state == ARGMAX) begin
            result_idx <= argmax_idx;
        end
    end

    //====================== 输出握手 ======================
    // WRITE_BACK 状态时，out_done 为 0 表示尚未输出
    // 当 data_out_tready 为高时，将 out_done 置 1，表示数据已发送
    // 之后 data_out_tvalid 将变为低
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            out_done <= 1'b0;
        end else if (state == IDLE) begin
            out_done <= 1'b0;
        end else if (state == WRITE_BACK && !out_done && data_out_tready) begin
            out_done <= 1'b1;
        end
    end

    // 输出数据: argmax 索引
    assign data_out_tdata = result_idx;
    // 输出有效: 在 WRITE_BACK 状态且尚未发送完成时有效
    assign data_out_tvalid = (state == WRITE_BACK && !out_done);

endmodule
