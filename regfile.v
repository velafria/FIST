module regfile #(
    parameter DATA_WIDTH = 472,
    parameter ADDR_WIDTH = 9,
    parameter DEPTH      = 484
)(
    input wire                  clk,
    input wire                  rst,

    input wire                  we,
    input wire [ADDR_WIDTH-1:0] wr_addr,
    input wire [DATA_WIDTH-1:0] wr_data,

    input wire [ADDR_WIDTH-1:0] rd_addr,
    output reg [DATA_WIDTH-1:0] rd_data
);

    reg [DATA_WIDTH-1:0] rg [DEPTH-1:0];

    always @(posedge clk) begin
        if (we) begin
            rg[wr_addr] <= wr_data;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            rd_data <= {DATA_WIDTH{1'b0}};
        end else begin
            rd_data <= rg[rd_addr];
        end
    end

endmodule