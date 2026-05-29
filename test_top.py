import os
from pathlib import Path

import torch
from torchvision import datasets
import numpy as np
from PIL import Image

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

def load_weight(name):
    BASE_DIR = Path(__file__).resolve().parent
    weight_path = BASE_DIR / "weights" / f"{name}.pth"

    weight = torch.load(weight_path, map_location="cpu").numpy()

    half_level = 7
    data_range = np.max(np.abs(weight))
    weight = np.round(weight/data_range*half_level).astype(np.int8)
    weight = np.clip(weight, -7, 7)

    return weight


def pack_weight_bytes(weight):
    row, col = weight.shape[0], weight.shape[1]
    bytes_per_col = (row * 4 + 7) // 8
    payload = bytearray()
    for j in range(col):
        packed = 0
        for i in range(row):
            packed |= (int(weight[i, j]) & 0xF) << (i * 4)
        payload.extend(packed.to_bytes(bytes_per_col, byteorder="little"))
    return bytes(payload)

def load_image(path=None):
    BASE_DIR = Path(__file__).resolve().parent
    image_path = Path(path) if path is not None else BASE_DIR / "image" / "test_7.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = Image.open(image_path).convert("L").resize((22, 22), resampling)
    image = np.asarray(image, dtype=np.float32)
    image = 255.0 - image

    image_max = np.max(np.abs(image))
    if image_max == 0:
        return np.zeros(484, dtype=np.uint8)

    image = np.round(image / image_max * 127)
    image = np.clip(image, 0, 127).astype(np.uint8)
    return image.reshape(-1)

def load_mnist_image(index=8478, is_train=False):
    """
    从 PyTorch 的 MNIST 数据集中读取指定索引的图片，
    并按照与 load_image 相同的逻辑进行缩放、量化和展平。
    
    参数:
    - index: 想要读取的图片索引 (0 ~ 59999)
    - is_train: True 表示读取训练集，False 表示读取测试集
    
    返回:
    - image_flat: 量化到 0~127 并且展平为 484 维的 np.uint8 数组
    - label: 该图片的正确数字标签 (int)
    """
    BASE_DIR = Path(__file__).resolve().parent
    data_dir = BASE_DIR / "data"
    
    mnist_dataset = datasets.MNIST(
        root=str(data_dir), 
        train=is_train, 
        download=True
    )
    
    pil_image, label = mnist_dataset[index]
    
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = pil_image.resize((22, 22), resampling)
    
    image = np.asarray(image, dtype=np.float32)
    
    image_max = np.max(np.abs(image))
    if image_max == 0:
        return np.zeros(484, dtype=np.uint8), int(label)

    image = np.round(image / image_max * 127)
    image = np.clip(image, 0, 127).astype(np.uint8)
    
    return image.reshape(-1), int(label)


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

    weight_fc1 = load_weight("fc1")
    weight_fc2 = load_weight("fc2")

    axis_source = AxiStreamSource(AxiStreamBus.from_prefix(dut, "data_in"), dut.clk, dut.rst)
    axis_sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "data_out"), dut.clk, dut.rst)

    hd_axis_monitor = axisMonitor(dut.clk, dut.hd_tvalid, dut.hd_tready, dut.hd_tdata)
    capture_hd = cocotb.start_soon(hd_axis_monitor.start())

    dut.en.value = 1
    await RisingEdge(dut.clk)

    image = load_image()
    
    #image, label = load_mnist_image()
    load_bytes = (
        pack_weight_bytes(weight_fc1)
        + pack_weight_bytes(weight_fc2)
        + bytes(image.astype(np.uint8).tolist())
    )

    await axis_source.send(load_bytes)

    await axis_source.wait()
    cocotb.log.info("Load weights and image into top module")

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


    expected_hidden_layer = weight_fc1.astype(np.int32) @ image.astype(np.int32)
    expected_hidden_layer[expected_hidden_layer < 0] = 0
    
    SHIFT = 16
    FC1_MUL = 1557
    ROUND = 1 << (SHIFT - 1)

    expected_hidden_layer_requant = (expected_hidden_layer * FC1_MUL + ROUND) >> SHIFT
    expected_hidden_layer_requant = np.clip(expected_hidden_layer_requant, 0, 127)

    #assert np.array_equal(received_data == hidden_layer_requant)
    #cocotb.log.info("fc1 test pass")

    expected_output_data = weight_fc2.astype(np.int32) @ expected_hidden_layer_requant.astype(np.int32)
    expected_result = int(np.argmax(expected_output_data))

    #assert np.array_equal(hd_layer_requant, expected_hidden_layer_requant)
    #assert np.array_equal(output_data, expected_output_data)
    assert result == expected_result
    #assert result == label
    cocotb.log.info(f"fc2 test pass, expect{expected_result}, get{result}")
    
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
        waves=True,
        always=True
    )

    runner.test(hdl_toplevel="top",
                test_module="test_top",
                waves=True,
                plusargs=["+fst", "+dumpfile=sim_build/top.fst"])

if __name__ == "__main__":
    test_runner()
