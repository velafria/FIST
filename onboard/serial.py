import serial
import time
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import datasets

# ==========================================
# 配置参数
# ==========================================
# 在 Mac 终端运行 ls /dev/cu.usbserial* 找到你的具体串口名并替换
SERIAL_PORT = '/dev/cu.usbserial-14101' 
BAUD_RATE = 115200
IMAGE_BYTES = 484

def load_image(path=None):
    """ 保持与你 cocotb 仿真完全一致的单张图片加载与量化逻辑 """
    BASE_DIR = Path(__file__).resolve().parent
    image_path = Path(path) if path is not None else BASE_DIR / "image" / "test_6.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"找不到输入图片: {image_path}")

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = Image.open(image_path).convert("L").resize((22, 22), resampling)
    image = np.asarray(image, dtype=np.float32)
    image = 255.0 - image

    image_max = np.max(np.abs(image))
    if image_max == 0:
        return np.zeros(IMAGE_BYTES, dtype=np.uint8)

    image = np.round(image / image_max * 127)
    image = np.clip(image, 0, 127).astype(np.uint8)
    return image.reshape(-1)

def load_mnist_image(index=8478, is_train=False):
    """ 保持与你 cocotb 仿真完全一致的 从 MNIST 数据集直接提取的逻辑 """
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
        return np.zeros(IMAGE_BYTES, dtype=np.uint8), int(label)

    image = np.round(image / image_max * 127)
    image = np.clip(image, 0, 127).astype(np.uint8)
    
    return image.reshape(-1), int(label)

def main():
    # 1. 准备要发送的图像数据
    try:
        # 选项 A: 读本地的 test_6.jpg (确保你 Mac 相应路径下有这张图)
        image_data = load_image()
        expected_label = "未知 (本地图片)"
        
        # 选项 B: 如果想用 MNIST 数据集里的图，取消下面这行的注释：
        # image_data, expected_label = load_mnist_image(index=8478)
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 转换成字节流
    send_bytes = bytes(image_data.tolist())
    if len(send_bytes) != IMAGE_BYTES:
        print(f"错误: 图像数据长度为 {len(send_bytes)} 字节，必须严格为 {IMAGE_BYTES} 字节！")
        return

    # 2. 串口通信
    print(f"正在尝试连接串口: {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=8) # 考虑到硬件计算和传输，设置8秒超时
        
        # 给板子一点稳定时间，或者丢弃掉之前的缓存
        time.sleep(0.5)
        ser.reset_input_buffer()
        
        print("--------------------------------------------------")
        print("请确保 ZYNQ 开发板已上电，且运行了 C 程序。")
        print("如果你是刚下载完程序，可能需要按下板子上的复位键，或者观察下方提示。")
        print("--------------------------------------------------")
        
        # 尝试读取 ZYNQ 打印出来的 "MNIST ready: send 484 raw image bytes."
        # 如果读取到了说明握手成功，若没读到我们也直接尝试发送
        if ser.in_waiting:
            init_msg = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"来自板子的提示: {init_msg.strip()}")

        print(f"-> 正在发送 484 字节图像数据 (预期标签: {expected_label})...")
        ser.write(send_bytes)
        ser.flush() # 确保数据完全发出
        
        print("<- 数据发送完毕，正在等待硬件加速器返回分类结果...")
        
        # 3. 接收返回的 1 字节分类数字
        result_byte = ser.read(1)
        
        if result_byte:
            predicted_digit = int.from_bytes(result_byte, byteorder='little')
            print("\n================================================")
            print(f"🎉 识别成功！")
            print(f"硬件预测数字 (Predicted): {predicted_digit}")
            print(f"标准参考数字 (Expected) : {expected_label}")
            print("================================================")
        else:
            print("\n 错误：超时未收到结果。可能原因：")
            print("1. PL端加速器卡死或没有产生 data_out_tvalid 信号。")
            print("2. AXI FIFO 读写逻辑未对齐。")
            
        # 捕获 C 语言中最后打印的 xil_printf("\r\nresult=%u\r\n", result)
        time.sleep(0.2)
        if ser.in_waiting:
            tail_msg = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            print(f"\n板子调试输出详情:\n{tail_msg.strip()}")
            
        ser.close()
        
    except serial.SerialException as e:
        print(f"串口错误: {e}. 请检查串口号是否被占用或连接断开。")

if __name__ == "__main__":
    main()