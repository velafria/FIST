module top (
    input  wire         clk,
    input  wire         rst,

    input  wire         en,

    input  wire [7:0]   data_in,
    input  wire         data_in_valid,
    output wire         data_in_ready
);

    wire [8:0]  w1_addr;
    wire [471:0] w1_data;

    wire [7:0]  hd_data;
    wire        hd_valid;
    wire        hd_ready;

    fc1 u_fc1 (
        .clk        (clk),
        .rst        (rst),

        .en         (en),
        
        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .data_in_ready(data_in_ready),

        .weight_addr(w1_addr),
        .weight_data_col(w1_data),

        .data_out(hd_data),
        .data_out_valid(hd_valid),
        .data_out_ready(hd_ready)
        
        
    );

    regfile #(
        .DATA_WIDTH(472),
        .ADDR_WIDTH(9),
        .DEPTH(484)
    ) w1_regfile (
        .clk        (clk),
        .rst        (rst),
        
        .we         (1'b0),
        .wr_addr    (9'b0),
        .wr_data    (472'b0),
        
        .rd_addr    (w1_addr),
        .rd_data    (w1_data)
    );

endmodule