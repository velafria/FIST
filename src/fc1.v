module fc1 (
    input  wire         clk,
    input  wire         rst,
    input  wire         en,

    input  wire signed [7:0]   data_in_tdata,
    input  wire         data_in_tvalid,
    output wire         data_in_tready,

    output reg  [8:0]   weight_addr,
    input  wire [471:0] weight_data_col,

    output wire signed [7:0]   data_out_tdata,
    output wire         data_out_tvalid,
    input  wire         data_out_tready
);

    localparam IDLE       = 2'b00;
    localparam CALC       = 2'b01;
    localparam RELU       = 2'b10;
    localparam WRITE_BACK = 2'b11;

    reg [1:0] state, next_state;
    reg [8:0] in_cnt;          // 输入计数器 (0~484)
    reg [6:0] out_cnt;         // 输出计数器
    reg [8:0] col_cnt;         // 乘累加列计数器 (0~484)

    reg signed [7:0] input_buf [0:483];    // 存储输入向量
    reg signed [19:0] acc [0:117];         // 累加器
    reg signed [7:0]  result [0:117];      // 最终结果

    wire signed [3:0] weight_col [0:117];
    genvar i;
    generate
        for (i = 0; i < 118; i = i + 1) begin
            assign weight_col[i] = $signed(weight_data_col[i*4+3 : i*4]);
        end
    endgenerate

    assign data_in_tready = (state == IDLE) || (state == CALC && in_cnt < 9'd484);

    // 状态机
    always @(posedge clk or posedge rst) begin
        if (rst) state <= IDLE;
        else state <= next_state;
    end

    always @(*) begin
        next_state = state;
        case (state)
            IDLE: if (en) next_state = CALC;
            CALC: if (in_cnt == 9'd484 && col_cnt == 9'd484) next_state = RELU;
            RELU: next_state = WRITE_BACK;
            WRITE_BACK: if (out_cnt == 7'd117 && data_out_tready) next_state = IDLE;
        endcase
    end

    // 输入接收阶段
    always @(posedge clk or posedge rst) begin
        if (rst) in_cnt <= 9'd0;
        else if (state == IDLE) in_cnt <= 9'd0;
        else if (state == CALC && data_in_tvalid && in_cnt < 9'd484) begin
            input_buf[in_cnt] <= data_in_tdata;
            in_cnt <= in_cnt + 9'd1;
        end
    end

    // 乘累加阶段：逐列输出地址，并累加
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            col_cnt <= 9'd0;
            weight_addr <= 9'd0;
        end else if (state == IDLE) begin
            col_cnt <= 9'd0;
            weight_addr <= 9'd0;
        end else if (state == CALC && in_cnt == 9'd484 && col_cnt < 9'd484) begin
            weight_addr <= col_cnt;                // 输出列地址
            col_cnt <= col_cnt + 9'd1;
        end
    end

    // 累加器更新（使用组合权重，因为 regfile 已改为组合输出）
    integer j;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 118; j = j + 1) acc[j] <= 20'd0;
        end else if (state == IDLE) begin
            for (j = 0; j < 118; j = j + 1) acc[j] <= 20'd0;
        end else if (state == CALC && in_cnt == 9'd484 && col_cnt > 0 && col_cnt <= 9'd484) begin
            // 当前周期的 weight_col 对应地址 col_cnt-1
            for (j = 0; j < 118; j = j + 1)
                acc[j] <= acc[j] + (weight_col[j] * input_buf[col_cnt-1]);
        end
    end

    // RELU + 量化
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (j = 0; j < 118; j = j + 1) result[j] <= 8'd0;
        end else if (state == RELU) begin
            for (j = 0; j < 118; j = j + 1) begin
                if (acc[j][19]) result[j] <= 8'd0;
                else if (acc[j] > 20'd127) result[j] <= 8'd127;
                else result[j] <= acc[j][7:0];
            end
        end
    end

    // 输出计数
    always @(posedge clk or posedge rst) begin
        if (rst) out_cnt <= 7'd0;
        else if (state == IDLE) out_cnt <= 7'd0;
        else if (state == WRITE_BACK && data_out_tready && out_cnt < 7'd117) out_cnt <= out_cnt + 7'd1;
    end

    assign data_out_tdata = result[out_cnt];
    assign data_out_tvalid = (state == WRITE_BACK);

endmodule
