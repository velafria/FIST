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
async def test_fc1(dut):
    cocotb.log.info(dir(dut))

    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weight_fc1 = load_weight(dut, "fc1")
    weight_fc2 = load_weight(dut, "fc2")
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

    '''
    all_frame = []
    for _ in range(118):
        frame = await axis_sink.recv()
        all_frame.append(np.frombuffer(frame.tdata, dtype=np.uint8))
    received_data = np.concatenate(all_frame)
    '''

    frame = await axis_sink.recv()
    result = int(np.frombuffer(frame.data, dtype=np.uint8))
    

    hidden_layer = weight_fc1 @ image
    hidden_layer[hidden_layer < 0] = 0
    
    SHIFT = 16
    FC1_MUL = 1557
    ROUND = 1 << (SHIFT - 1)

    hidden_layer_requant = (hidden_layer * FC1_MUL + ROUND) >> SHIFT
    hidden_layer_requant = np.clip(hidden_layer_requant, 0, 127)

    #assert np.array_equal(received_data == hidden_layer_requant)
    #cocotb.log.info("fc1 test pass")

    output_data = weight_fc2 @ hidden_layer_requant
    expected_result = int(np.argmax(output_data))

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