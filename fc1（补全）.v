module fc1 (
    //时钟，重置，使能
    input  wire         clk,
    input  wire         rst,
    input  wire         en,
    
    // 输入流接口
    input  wire signed [7:0]   data_in,       // 串行输入 484 个 int8 (有符号)
    input  wire         data_in_valid,        // 输入完毕
    output wire         data_in_ready,        // 准备接收

    output reg  [8:0]   weight_addr,     // 矩阵列地址 (0-483，需要 9 bits)
    input  wire [471:0] weight_data_col, // 读到的整列权重 (118 * 4bit = 472 bits)

    // 输出数据流 (发往激活函数或下一层)
    output wire signed [7:0]   data_out,      // 串行输出 118 个 int8 (有符号)
    output wire         data_out_valid,       // 输出准备完毕
    input  wire         data_out_ready        // 接收端准备接收
);

    // 状态定义
    localparam IDLE       = 2'b00;          // 空闲
    localparam CALC       = 2'b01;          // 计算
    localparam RELU       = 2'b10;          // 量化
    localparam WRITE_BACK = 2'b11;          // 读取
    
    reg [1:0] state, next_state;            //状态转移辅助
    
    // 计数器定义
    reg [8:0] in_cnt;      // 输入计数器 0-483 (9 bits)
    reg [6:0] out_cnt;     // 输出计数器 0-117 (7 bits)
    
    // 累加器定义 (118个有符号累加器)
    // 20-bit足够：8-bit*4-bit=12-bit，484次累加最多需要20-bit
    reg signed [19:0] acc [0:117];
    
    // 输出寄存器 (118个有符号8-bit结果)
    reg signed [7:0] result [0:117];
    
    // 流水线寄存器
    reg signed [7:0] data_in_reg;           // 延迟一拍的输入数据（有符号）
    reg signed [3:0] weight_reg [0:117];    // 延迟一拍的权重数组（有符号）
    reg calc_valid;                          // 乘累加有效标志
    
    // 从472-bit总线提取118个4-bit有符号权重
    wire signed [3:0] weight_col [0:117];
    genvar i;
    generate
        for (i = 0; i < 118; i = i + 1) begin
            // 提取4-bit权重，并解释为有符号数
            assign weight_col[i] = $signed(weight_data_col[i*4+3 : i*4]);
        end
    endgenerate
    
    // data_in_ready: 只在IDLE或CALC状态时ready
    assign data_in_ready = (state == IDLE) || (state == CALC);
    
    // ========== 状态机时序逻辑 ==========
    always @(posedge clk or posedge rst) begin
        if (rst)
            state <= IDLE;          // 重置时转空闲
        else
            state <= next_state;    // 上升沿触发状态转移
    end
    
    // ========== 状态机组合逻辑 ==========
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (en && data_in_valid)
                    next_state = CALC;      // 使能且数据输入完毕时转移状态
            end
            
            CALC: begin
                if (in_cnt == 9'd483 && calc_valid)
                    next_state = RELU;      // 处理完第484列后转移状态
            end
            
            RELU: begin
                next_state = WRITE_BACK;    // ReLU和量化在1个周期内完成，直接转移
            end
            
            WRITE_BACK: begin
                if (out_cnt == 7'd117 && data_out_ready)
                    next_state = IDLE;      // 输出完毕且接收端准备完毕后状态转移
            end
        endcase
    end
    
    // ========== 输入计数器逻辑 ==========
    always @(posedge clk or posedge rst) begin
        if (rst)
            in_cnt <= 9'd0;         // 重置时置零
        else if (state == IDLE)
            in_cnt <= 9'd0;         // 空闲时置零
        else if (state == CALC && data_in_valid) begin
            if (in_cnt == 9'd483)
                in_cnt <= 9'd0;     // 循环计数结束后置零
            else
                in_cnt <= in_cnt + 9'd1;    // 每个周期结束后加1
        end
    end
    
    // ========== 输出计数器逻辑 ==========
    always @(posedge clk or posedge rst) begin
        if (rst)
            out_cnt <= 7'd0;        // 重置时置零
        else if (state == IDLE)
            out_cnt <= 7'd0;        // 空闲时置零
        else if (state == WRITE_BACK && data_out_ready) begin
            if (out_cnt == 7'd117)
                out_cnt <= 7'd0;    //循环计数结束后置零
            else
                out_cnt <= out_cnt + 7'd1;  // 每个周期结束后加1
        end
    end
    
    // ========== 权重地址生成 ==========
    // 每次请求矩阵的第k列（k = in_cnt）
    always @(posedge clk or posedge rst) begin
        if (rst)
            weight_addr <= 9'd0;        // 重置时置零
        else if (state == CALC && data_in_valid)
            weight_addr <= in_cnt;  // 请求第in_cnt列
    end
    
    // ========== 输入数据延迟寄存器 ==========
    // 将当前周期的data_in延迟一拍，与权重数据对齐
    always @(posedge clk or posedge rst) begin
        if (rst)
            data_in_reg <= 8'd0;        // 重置时置零
        else if (state == CALC && data_in_valid)
            data_in_reg <= data_in;  // 存储当前输入的向量元素
    end
    
    // ========== 权重延迟寄存器 ==========
    // 将当前周期的权重列延迟一拍，与输入数据对齐
    integer j;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 118; j = j + 1)
                weight_reg[j] <= 4'd0;        // 重置时置零
        end
        else if (state == CALC) begin
            // 存储当前读到的整列权重
            for (j = 0; j < 118; j = j + 1)
                weight_reg[j] <= weight_col[j];
        end
    end
    
    // ========== 乘累加有效标志 ==========
    // 表示当前周期是否需要进行乘累加运算
    always @(posedge clk or posedge rst) begin
        if (rst)
            calc_valid <= 1'b0;  // 重置时置零
        else if (state == CALC && data_in_valid)
            calc_valid <= 1'b1;  // 数据有效后的下一个周期进行乘累加
        else
            calc_valid <= 1'b0;
    end
    
    // ========== 核心计算：并行乘累加 ==========
    // 计算顺序：每个周期内，用向量各行与矩阵当前列各行相乘，结果累加到输出向量上，所有列计算结束后输出结果
    // 每个周期：acc[j] += weight_reg[j] * data_in_reg
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 118; j = j + 1)
                acc[j] <= 20'd0;       // 重置时置零
        end
        else if (state == IDLE) begin
            // IDLE状态清零累加器，准备下一次计算
            for (j = 0; j < 118; j = j + 1)
                acc[j] <= 20'd0;
        end
        else if (state == CALC && calc_valid) begin
            // 并行执行118次有符号乘累加
            for (j = 0; j < 118; j = j + 1) begin
                acc[j] <= acc[j] + (weight_reg[j] * data_in_reg);
            end
        end
    end
    
    // ========== ReLU激活 + 饱和量化 ==========
    // 此部分暂时未验证！！！
    // 将有符号20-bit累加结果转换为有符号8-bit输出
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 118; j = j + 1)
                result[j] <= 8'd0;
        end
        else if (state == RELU) begin
            for (j = 0; j < 118; j = j + 1) begin
                // ReLU激活函数：负数归零
                if (acc[j][19] == 1'b1)
                    result[j] <= 8'd0;
                else begin
                    // 饱和量化到int8范围 [0, 127]
                    if (acc[j] > 20'd127)
                        result[j] <= 8'd127;
                    else
                        result[j] <= acc[j][7:0];
                end
            end
        end
    end
    
    // ========== 输出数据选择 ==========
    // 根据out_cnt从118个结果中选择一个输出
    assign data_out = result[out_cnt];
    
    // ========== 输出有效信号 ==========
    assign data_out_valid = (state == WRITE_BACK);
    
endmodule