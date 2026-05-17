module fc1 (
    input  wire         clk,
    input  wire         rst,
    input  wire         en,
    
    // 输入数据流 (来自上一层或 Input Buffer)
    input  wire [7:0]   data_in,       // 串行输入 484 个 int8
    input  wire         data_in_valid,
    output wire         data_in_ready,

    output reg  [8:0]   weight_addr,     // 权重列地址 (0-483，需要 9 bits)
    input  wire [471:0] weight_data_col, // 读到的整列权重 (118 * 4bit = 472 bits)

    // 输出数据流 (发往激活函数或下一层)
    output wire [7:0]   data_out,      // 串行输出 118 个 int8
    output wire         data_out_valid,
    input  wire         data_out_ready,

);
endmodule