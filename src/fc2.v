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

endmodule