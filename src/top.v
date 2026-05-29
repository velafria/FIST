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

    wire [8:0]  w1_addr;
    wire [471:0] w1_data;

    wire [6:0]  w2_addr;
    wire [39:0] w2_data;

    wire [7:0]  hd_tdata;
    wire        hd_tvalid;
    wire        hd_tready;

    localparam TOP_IDLE    = 3'd0;
    localparam TOP_LOAD_W1 = 3'd1;
    localparam TOP_LOAD_W2 = 3'd2;
    localparam TOP_START   = 3'd3;
    localparam TOP_INFER   = 3'd4;

    localparam W1_BYTES_PER_COL = 6'd59;
    localparam W2_BYTES_PER_COL = 3'd5;

    reg [2:0] state;

    reg [8:0] w1_wr_addr;
    reg [5:0] w1_byte_cnt;
    reg [471:0] w1_shift;

    reg [6:0] w2_wr_addr;
    reg [2:0] w2_byte_cnt;
    reg [39:0] w2_shift;

    wire fc1_data_in_tready;
    wire load_w1_accept;
    wire load_w2_accept;
    wire [471:0] w1_wr_data_next;
    wire [39:0] w2_wr_data_next;
    wire w1_we;
    wire w2_we;

    assign data_in_tready = (state == TOP_LOAD_W1) ? 1'b1 :
                            (state == TOP_LOAD_W2) ? 1'b1 :
                            (state == TOP_INFER)   ? fc1_data_in_tready :
                                                     1'b0;

    assign load_w1_accept = (state == TOP_LOAD_W1) && data_in_tvalid && data_in_tready;
    assign load_w2_accept = (state == TOP_LOAD_W2) && data_in_tvalid && data_in_tready;

    assign w1_wr_data_next = w1_shift | ({464'd0, data_in_tdata} << (w1_byte_cnt * 8));
    assign w2_wr_data_next = w2_shift | ({32'd0, data_in_tdata} << (w2_byte_cnt * 8));

    assign w1_we = load_w1_accept && (w1_byte_cnt == W1_BYTES_PER_COL - 1'b1);
    assign w2_we = load_w2_accept && (w2_byte_cnt == W2_BYTES_PER_COL - 1'b1);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= TOP_IDLE;
            w1_wr_addr <= 9'd0;
            w1_byte_cnt <= 6'd0;
            w1_shift <= 472'd0;
            w2_wr_addr <= 7'd0;
            w2_byte_cnt <= 3'd0;
            w2_shift <= 40'd0;
        end else begin
            case (state)
                TOP_IDLE: begin
                    w1_wr_addr <= 9'd0;
                    w1_byte_cnt <= 6'd0;
                    w1_shift <= 472'd0;
                    w2_wr_addr <= 7'd0;
                    w2_byte_cnt <= 3'd0;
                    w2_shift <= 40'd0;
                    if (en) state <= TOP_LOAD_W1;
                end

                TOP_LOAD_W1: begin
                    if (load_w1_accept) begin
                        if (w1_byte_cnt == W1_BYTES_PER_COL - 1'b1) begin
                            w1_shift <= 472'd0;
                            w1_byte_cnt <= 6'd0;
                            if (w1_wr_addr == 9'd483) begin
                                w1_wr_addr <= 9'd0;
                                state <= TOP_LOAD_W2;
                            end else begin
                                w1_wr_addr <= w1_wr_addr + 9'd1;
                            end
                        end else begin
                            w1_shift <= w1_wr_data_next;
                            w1_byte_cnt <= w1_byte_cnt + 6'd1;
                        end
                    end
                end

                TOP_LOAD_W2: begin
                    if (load_w2_accept) begin
                        if (w2_byte_cnt == W2_BYTES_PER_COL - 1'b1) begin
                            w2_shift <= 40'd0;
                            w2_byte_cnt <= 3'd0;
                            if (w2_wr_addr == 7'd117) begin
                                w2_wr_addr <= 7'd0;
                                state <= TOP_START;
                            end else begin
                                w2_wr_addr <= w2_wr_addr + 7'd1;
                            end
                        end else begin
                            w2_shift <= w2_wr_data_next;
                            w2_byte_cnt <= w2_byte_cnt + 3'd1;
                        end
                    end
                end

                TOP_START: begin
                    state <= TOP_INFER;
                end

                TOP_INFER: begin
                    if (!en) state <= TOP_IDLE;
                end
            endcase
        end
    end

    fc1 u_fc1 (
        .clk        (clk),
        .rst        (rst),

        .en         ((state == TOP_START) || (state == TOP_INFER)),
        
        .data_in_tdata(data_in_tdata),
        .data_in_tvalid((state == TOP_INFER) && data_in_tvalid),
        .data_in_tready(fc1_data_in_tready),

        .weight_addr(w1_addr),
        .weight_data_col(w1_data),

        .data_out_tdata(hd_tdata),
        .data_out_tvalid(hd_tvalid),
        .data_out_tready(hd_tready)
        
        
    );

    fc2 u_fc2 (
        .clk        (clk),
        .rst        (rst),

        .en         ((state == TOP_START) || (state == TOP_INFER)),
        
        .data_in_tdata(hd_tdata),
        .data_in_tvalid(hd_tvalid),
        .data_in_tready(hd_tready),

        .weight_addr(w2_addr),
        .weight_data_col(w2_data),

        .data_out_tdata(data_out_tdata),
        .data_out_tvalid(data_out_tvalid),
        .data_out_tready(data_out_tready)
        
        
    );

    regfile #(
        .DATA_WIDTH(472),
        .ADDR_WIDTH(9),
        .DEPTH(484)
    ) regfile_w1 (
        .clk        (clk),
        .rst        (rst),
        
        .we         (w1_we),
        .wr_addr    (w1_wr_addr),
        .wr_data    (w1_wr_data_next),
        
        .rd_addr    (w1_addr),
        .rd_data    (w1_data)
    );

    regfile #(
        .DATA_WIDTH(40),
        .ADDR_WIDTH(7),
        .DEPTH(118)
    ) regfile_w2 (
        .clk        (clk),
        .rst        (rst),
        
        .we         (w2_we),
        .wr_addr    (w2_wr_addr),
        .wr_data    (w2_wr_data_next),
        
        .rd_addr    (w2_addr),
        .rd_data    (w2_data)
    );

endmodule
