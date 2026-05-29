import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_pass(x):
    y = torch.round(x)
    return (y - x).detach() + x


def fake_quant_symmetric_per_layer(x, bit, fixed_scale = None, eps = 1e-12):
    quant_max = 2 ** (bit - 1) - 1
    quant_min = -quant_max
    if fixed_scale is None:
        scale = x.detach().abs().max() / quant_max
    else:
        scale = torch.as_tensor(fixed_scale, device = x.device, dtype = x.dtype)
    scale = torch.clamp(scale, min = eps)
    q = torch.clamp(round_pass(x / scale), quant_min, quant_max)
    return q * scale, scale


def fake_quant_unsigned_per_tensor(x, bit, fixed_scale = None, eps = 1e-12):
    quant_max = 2 ** bit - 1
    if fixed_scale is None:
        scale = x.detach().max() / quant_max
    else:
        scale = torch.as_tensor(fixed_scale, device = x.device, dtype = x.dtype)
    scale = torch.clamp(scale, min = eps)
    q = torch.clamp(round_pass(x / scale), 0, quant_max)
    return q * scale, scale


def fake_quant_unsigned_per_sample(x, bit, eps = 1e-12):
    quant_max = 2 ** bit - 1
    flat = x.detach().reshape(x.shape[0], -1)
    scale = flat.max(dim = 1).values / quant_max
    scale = torch.clamp(scale, min = eps).view(x.shape[0], *([1] * (x.dim() - 1)))
    q = torch.clamp(round_pass(x / scale), 0, quant_max)
    return q * scale, scale


class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_bit = 4, bias = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.weight_bit = weight_bit

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        weight_q, _ = fake_quant_symmetric_per_layer(self.linear.weight, self.weight_bit)
        return F.linear(x, weight_q, self.linear.bias)


class SimpleFCModel(nn.Module):
    def __init__(self, fc1_scale = None):
        super().__init__()
        self.resize = nn.AdaptiveAvgPool2d((22, 22))
        self.fc1 = QuantLinear(22 * 22, 118, weight_bit = 4, bias = False)
        self.fc2 = QuantLinear(118, 10, weight_bit = 4, bias = False)
        self.fc1_scale = fc1_scale

    def set_fc1_scale(self, scale):
        self.fc1_scale = float(scale)

    def forward(self, x, return_fc1_relu = False):
        # Hardware input encoding: each image is quantized to 0..127.
        x = self.resize(x)
        x, _ = fake_quant_unsigned_per_sample(x, bit = 7)
        x = x.view(-1, 22 * 22)

        fc1 = self.fc1(x)
        fc1_relu = F.relu(fc1)
        if return_fc1_relu:
            return fc1_relu

        fc1_q, _ = fake_quant_unsigned_per_tensor(fc1_relu, bit = 7, fixed_scale = self.fc1_scale)
        return self.fc2(fc1_q)


def check_accuracy(loader, model, device):
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if was_training:
        model.train()
    return 100 * correct / total


def train_one_epoch(loader, model, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        cumulative_accuracy = 100 * correct / total
        print(f"\rEpoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(loader)}], "
              f"Loss: {loss.item():.4f}, Cumulative Accuracy: {cumulative_accuracy:.2f}%",
              end = '', flush = True)
    print('')
    return running_loss / len(loader)


def calibrate_fc1_scale(loader, model, device, percentile = 99.5):
    was_training = model.training
    model.eval()
    values = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            fc1_relu = model(images, return_fc1_relu = True)
            values.append(fc1_relu.detach().flatten().cpu())
    if was_training:
        model.train()

    all_values = torch.cat(values)
    clip_value = torch.quantile(all_values, percentile / 100.0).item()
    if clip_value <= 0:
        clip_value = all_values.max().item()
    scale = clip_value / 127.0
    return scale, clip_value


def quantize_weight_int4(weight):
    scale = weight.detach().abs().max() / 7.0
    scale = torch.clamp(scale, min = 1e-12)
    q = torch.round(weight.detach().cpu() / scale.cpu()).clamp(-7, 7).to(torch.int8)
    return q, float(scale.cpu())


def save_hardware_artifacts(model, directory = "./weights"):
    os.makedirs(directory, exist_ok = True)

    fc1_q, fc1_weight_scale = quantize_weight_int4(model.fc1.weight)
    fc2_q, fc2_weight_scale = quantize_weight_int4(model.fc2.weight)

    torch.save(fc1_q, os.path.join(directory, "fc1_int4.pth"))
    torch.save(fc2_q, os.path.join(directory, "fc2_int4.pth"))
    torch.save(model.fc1.weight.detach().cpu(), os.path.join(directory, "fc1.pth"))
    torch.save(model.fc2.weight.detach().cpu(), os.path.join(directory, "fc2.pth"))

    scales = {
        "input_storage_bit": 8,
        "input_quant_max": 127,
        "input_quant": "per_image_unsigned_0_to_127",
        "weight_bit": 4,
        "weight_quant_min": -7,
        "weight_quant_max": 7,
        "weight_quant": "per_layer_symmetric_signed_-7_to_7",
        "fc1_output_storage_bit": 8,
        "fc1_output_quant_max": 127,
        "fc1_output_quant": "fixed_unsigned_0_to_127",
        "fc1_output_scale": float(model.fc1_scale),
        "fc1_weight_scale": fc1_weight_scale,
        "fc2_weight_scale": fc2_weight_scale,
        "fc1_pth_shape": list(fc1_q.shape),
        "fc2_pth_shape": list(fc2_q.shape),
    }
    with open(os.path.join(directory, "scales.json"), "w") as f:
        json.dump(scales, f, indent = 2)
    torch.save(scales, os.path.join(directory, "scales.pth"))
    return scales
