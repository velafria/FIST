`timescale 1ns/1ps

module top (
    input  wire         clk,
    input  wire         rst,

    input  wire         en,

    input  wire [7:0]   data_in_tdata,
    input  wire         data_in_tvalid,
    output wire         data_in_tready,

    output wire [7:0]   data_out_tdata,
    output wire         data_out_tvalid,
    input  wire         data_out_tready
);
    assign data_out_tdata = hd_tdata;
    assign data_out_tvalid = hd_tvalid;
    assign hd_tready = data_out_tready;

    wire [8:0]  w1_addr;
    wire [471:0] w1_data;

    wire [7:0]  hd_tdata;
    wire        hd_tvalid;
    wire        hd_tready;

    fc1 u_fc1 (
        .clk        (clk),
        .rst        (rst),

        .en         (en),
        
        .data_in_tdata(data_in_tdata),
        .data_in_tvalid(data_in_tvalid),
        .data_in_tready(data_in_tready),

        .weight_addr(w1_addr),
        .weight_data_col(w1_data),

        .data_out_tdata(hd_tdata),
        .data_out_tvalid(hd_tvalid),
        .data_out_tready(hd_tready)
        
        
    );

    regfile #(
        .DATA_WIDTH(472),
        .ADDR_WIDTH(9),
        .DEPTH(484)
    ) regfile_w1 (
        .clk        (clk),
        .rst        (rst),
        
        .we         (1'b0),
        .wr_addr    (9'b0),
        .wr_data    (472'b0),
        
        .rd_addr    (w1_addr),
        .rd_data    (w1_data)
    );

endmodule