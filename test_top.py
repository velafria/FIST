import os
from pathlib import Path

import torch
import numpy as np

import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
from cocotb_tools.runner import get_runner
from cocotb.handle import Immediate
from cocotbext.axi import AxiStreamBus, AxiStreamSource, AxiStreamSink
from cocotb.queue import Queue

class axisMonitor:
    def __init__(self, clk, tvalid, tready, tdata, name="Monitor"):
        self.clk = clk
        self.tvalid = tvalid
        self.tready = tready
        self.tdata = tdata
        self.name = name
        self.queue = Queue()
        self.active = False
    
    async def start(self):
        self.active = True
        while self.active:
            await RisingEdge(self.clk)
            if self.tvalid.value == 1 and self.tready.value == 1:
                captured_val = self.tdata.value
                self.queue.put_nowait(int(captured_val))

    def stop(self):
        self.active = False

def load_weight(dut, name):
    BASE_DIR = Path(__file__).resolve().parent
    weight_path = BASE_DIR / "weights" / f"{name}.pth"

    weight = torch.load(weight_path, map_location="cpu").numpy()

    half_level = 7
    data_range = np.max(np.abs(weight))
    weight = np.round(weight/data_range*half_level).astype(np.int8)
    weight = np.clip(weight, -7, 7)

    row, col = weight.shape[0], weight.shape[1]

    regfile = getattr(dut, f"regfile_w{name[-1]}")

    for j in range(col):
        packed = 0
        for i in range(row):
            packed |= (int(weight[i, j]) & 0xF) << (i * 4)
        regfile.rg[j].value = packed

    return weight

def load_image(path=None):
    rng = np.random.default_rng(seed=51)
    image = rng.integers(0, 128, size=484)
    return image

async def reset(dut):
    dut.rst.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    cocotb.log.debug("Reset complete")

@cocotb.test()
async def test_top(dut):
    cocotb.log.info(dir(dut))

    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weight_fc1 = load_weight(dut, "fc1")
    weight_fc2 = load_weight(dut, "fc2")
    cocotb.log.info("Load weight into regfile directly")

    axis_source = AxiStreamSource(AxiStreamBus.from_prefix(dut, "data_in"), dut.clk, dut.rst)
    axis_sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "data_out"), dut.clk, dut.rst)

    hd_axis_monitor = axisMonitor(dut.clk, dut.hd_tvalid, dut.hd_tready, dut.hd_tdata)
    capture_hd = cocotb.start_soon(hd_axis_monitor.start())

    dut.en.value = 1
    await RisingEdge(dut.clk)

    image = load_image()
    load_bytes = bytes(image.astype(np.uint8).tolist())

    await axis_source.send(load_bytes)

    await axis_source.wait()
    cocotb.log.info("Load image into top module")

    while int(dut.u_fc2.state.value) != 2:  # ARGMAX = 2
        await RisingEdge(dut.clk)

    output_data = np.array(
        [dut.u_fc2.acc[i].value.to_signed() for i in range(10)],
        dtype=np.int32,
    )

    frame = await axis_sink.recv()
    result = frame.tdata[0]
    
    hd_axis_monitor.stop()
    await capture_hd

    hd_layer_requant = []
    while not hd_axis_monitor.queue.empty():
        hd_layer_requant.append(hd_axis_monitor.queue.get_nowait())
    hd_layer_requant = np.array(hd_layer_requant, dtype=np.uint8)

    #output_data = 


    expected_hidden_layer = weight_fc1 @ image
    expected_hidden_layer[expected_hidden_layer < 0] = 0
    
    SHIFT = 16
    FC1_MUL = 1557
    ROUND = 1 << (SHIFT - 1)

    expected_hidden_layer_requant = (expected_hidden_layer * FC1_MUL + ROUND) >> SHIFT
    expected_hidden_layer_requant = np.clip(expected_hidden_layer_requant, 0, 127)

    #assert np.array_equal(received_data == hidden_layer_requant)
    #cocotb.log.info("fc1 test pass")

    expected_output_data = weight_fc2 @ expected_hidden_layer_requant
    expected_result = int(np.argmax(expected_output_data))

    assert np.array_equal(hd_layer_requant, expected_hidden_layer_requant)
    assert np.array_equal(output_data, expected_output_data)
    assert result == expected_result
    cocotb.log.info("fc2 test pass")
    
def test_runner():
    sim = os.getenv("SIM", "icarus")

    proj_path = Path(__file__).resolve().parent / "src"

    sources = [proj_path / "top.v",
               proj_path / "fc1.v",
               proj_path / "fc2.v",
               proj_path / "regfile.v"]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="top",
        waves=True
    )

    runner.test(hdl_toplevel="top",
                test_module="test_top",
                waves=True,
                plusargs=["+fst", "+dumpfile=sim_build/top.fst"])

if __name__ == "__main__":
    test_runner()