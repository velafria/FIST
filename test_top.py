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

def load_weight(dut, name):
    BASE_DIR = Path(__file__).resolve().parent
    weight_path = BASE_DIR / "weights" / f"{name}.pth"

    weight = torch.load(weight_path, map_location="cpu").numpy()

    half_level = 7
    data_range = np.max(np.abs(weight))
    weight = np.round(weight/data_range*half_level).astype(np.int8)
    weight = np.clip(weight, -8, 7)

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
async def test_fc1(dut):
    cocotb.log.info(dir(dut))

    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weight = load_weight(dut, "fc1")
    cocotb.log.info("Load weight into regfile directly")

    axis_source = AxiStreamSource(AxiStreamBus.from_prefix(dut, "data_in"), dut.clk, dut.rst)
    axis_sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "data_out"), dut.clk, dut.rst)

    dut.en.value = 1
    await RisingEdge(dut.clk)

    image = load_image()
    load_bytes = bytes(image.astype(np.uint8).tolist())

    await axis_source.send(load_bytes)

    await axis_source.wait()
    cocotb.log.info("Load image into top module")

    all_frame = []
    for _ in range(118):
        frame = await axis_sink.recv()
        all_frame.append(np.frombuffer(frame.tdata, dtype=np.uint8))
    
    received_data = np.concatenate(all_frame)

    expected_data = weight @ image
    expected_data[expected_data < 0] = 0
    expected_data[expected_data > 127] = 127

    cocotb.log.info("Receive result from fc1")

    assert np.array_equal(received_data, expected_data)

    cocotb.log.info("fc1 test successful")
    
def test_runner():
    sim = os.getenv("SIM", "icarus")

    proj_path = Path(__file__).resolve().parent

    sources = [proj_path / "top.v",
               proj_path / "fc1.v",
               proj_path / "regfile.v"]

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="top",
        waves=True
    )

    runner.test(hdl_toplevel="top", test_module="test_top")

if __name__ == "__main__":
    test_runner()