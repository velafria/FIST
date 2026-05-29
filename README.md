# FIST — FPGA Inference for MNIST with Quantized MLP

A 2-layer MLP for MNIST digit classification, implemented in Verilog and co-simulated with cocotb. The model is trained with quantization-aware training (QAT) to produce INT4 weights and INT8 activations that map directly onto fixed-point hardware arithmetic.

## Architecture

```
Input (22×22, uint8 0–127)
    │  484 pixels via AXI-Stream
    ▼
  FC1  [118 × 484, INT4 weights]
    │  output-stationary MAC, ReLU, requantize → 118 × uint8
    ▼
  FC2  [10 × 118, INT4 weights]
    │  output-stationary MAC, argmax
    ▼
 Result (0–9) via AXI-Stream
```

**FC1 requantization formula** (hardware-friendly fixed-point):
```
output = clamp( (relu(acc) × 1557 + 32768) >> 16, 0, 127 )
```
The multiplier `1557` and shift `16` encode the activation scale calibrated during training. These must be updated whenever the model is retrained.

## Hardware Interface

The top module uses a single AXI-Stream input port. Data is streamed in the following order:

1. FC1 weights — 484 columns × 59 bytes each (118 × INT4 packed per column)
2. FC2 weights — 118 columns × 5 bytes each (10 × INT4 packed per column)
3. Input image — 484 bytes (one uint8 per pixel)

The predicted digit (0–9) is returned as a single byte on the AXI-Stream output.

| Signal | Direction | Description |
|---|---|---|
| `clk` | in | Clock |
| `rst` | in | Synchronous reset (active high) |
| `en` | in | Enable — hold high to start loading |
| `data_in_t*` | in | AXI-Stream input (weights + image) |
| `data_out_t*` | out | AXI-Stream output (result byte) |

## File Structure

```
src/
  top.v       — top-level: AXI-Stream loader, state machine, regfile instantiation
  fc1.v       — FC1 layer: MAC, ReLU, fixed-point requantize, AXI-Stream output
  fc2.v       — FC2 layer: MAC, argmax, AXI-Stream output
  regfile.v   — synchronous register file for weight storage
weights/
  fc1_int4.pth    — INT4 weight tensor [118, 484]
  fc2_int4.pth    — INT4 weight tensor [10, 118]
  fc1.pth / fc2.pth — float weights for analysis or training resume
  scales.json     — quantization parameters from training
train.py        — QAT training script
test_top.py     — cocotb testbench
utils.py        — model definition, QAT utilities, weight export
```

## Usage

### 1. Train the model

```bash
python train.py
```

This runs three stages:
1. QAT pretrain with dynamic FC1 activation scale (30 epochs)
2. Calibrate a fixed FC1 activation scale from the training set
3. Fine-tune with the fixed scale (5 epochs)

Outputs are saved to `weights/`. The calibrated `fc1_output_scale` in `scales.json` determines the requantization constants.

### 2. Update requantization constants

After training, `weights/scales.json` contains `fc1_output_scale`. Use it to derive the hardware constants:

```python
import json

with open("weights/scales.json") as f:
    scales = json.load(f)

SHIFT = 16                                          # keep at 16 unless more precision is needed
K1 = scales["fc1_weight_scale"] / (127 * scales["fc1_output_scale"])
SCALE_MUL = round(K1 * 2**SHIFT)
ROUND = 1 << (SHIFT - 1)                           # always 2^(SHIFT-1)

print(f"SCALE_MUL = {SCALE_MUL}")
print(f"SHIFT     = {SHIFT}")
print(f"ROUND     = {ROUND}")
```

The derivation: the integer accumulator relates to the float activation as `acc_float = acc_int × (image_max/127) × fc1_weight_scale`. Approximating `image_max ≈ 1` for normalized inputs, the requantization target `out_q = round(relu(acc_float) / fc1_output_scale)` becomes:

```
out_q = round(relu(acc_int) × K1)
K1    = fc1_weight_scale / (127 × fc1_output_scale)
```

`SCALE_MUL / 2^SHIFT` approximates `K1` in fixed-point integer arithmetic.

Then update the constants in two places:

**`src/fc1.v`, lines 150–152:**
```verilog
localparam signed [31:0] SCALE_MUL = 32'd1557;   // ← replace with new SCALE_MUL
localparam integer       SHIFT     = 16;          // ← replace if changed
localparam signed [31:0] ROUND     = 32'd32768;   // ← always 2^(SHIFT-1)
```

**`test_top.py`, lines 187–189:**
```python
SHIFT   = 16      # ← replace if changed
FC1_MUL = 1557    # ← replace with new SCALE_MUL
ROUND   = 1 << (SHIFT - 1)
```

The default values (`SCALE_MUL = 1557`, `SHIFT = 16`) were derived from `fc1_weight_scale = 0.1188` and `fc1_output_scale = 0.0394`, giving `K1 ≈ 0.02377`.

### 3. Run simulation

Requires [cocotb](https://www.cocotb.org/), [cocotb-tools](https://github.com/cocotb/cocotb-tools), [cocotbext-axi](https://github.com/alexforencich/cocotbext-axi), and [Icarus Verilog](http://iverilog.icarus.com/).

```bash
python test_top.py
```

The testbench streams FC1 weights, FC2 weights, and a test image into the DUT, then checks the argmax output against a Python reference model.

To use a different input, edit `test_top.py` and switch between `load_image()` (JPEG file from `image/`) and `load_mnist_image(index)` (PyTorch MNIST dataset).

**Preprocessing your own image:** `quant_image.py` converts an external JPEG to the format the hardware expects — grayscale, inverted (white-on-black), resized to 22×22, and quantized to uint8 0–127. Run it standalone to inspect the result:

```bash
python quant_image.py
```

Edit `INPUT_IMG_PATH` at the bottom of the file to point to your image. The processed array is what gets streamed into the DUT as the input image.

## Quantization Details

| Tensor | Format | Range |
|---|---|---|
| Input pixels | uint8 | 0 – 127 |
| FC1 / FC2 weights | INT4 | −7 – 7 |
| FC1 hidden activations | uint8 | 0 – 127 |
| FC2 accumulators | int32 | unbounded |

Weights are quantized per-layer symmetric. The input is quantized per-image unsigned. The FC1 output scale is fixed at calibration time and baked into the hardware as a multiply-shift pair.

## Dependencies

```
torch
torchvision
numpy
Pillow
cocotb
cocotb-tools
cocotbext-axi
iverilog
```
