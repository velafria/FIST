import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from utils import (
    SimpleFCModel,
    calibrate_fc1_scale,
    check_accuracy,
    save_hardware_artifacts,
    train_one_epoch,
)


# ====================== #
# Training configuration
# ====================== #
batch_size = 32
pretrain_epochs = 30
finetune_epochs = 5
pretrain_lr = 1e-4
finetune_lr = 5e-5
calibration_percentile = 99.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = os.path.dirname(os.path.abspath(__file__))


# ====================== #
# Dataset
# ====================== #
transform = transforms.ToTensor()
trainset = MNIST(root = os.path.join(base_dir, "data"), train = True, download = True, transform = transform)
testset = MNIST(root = os.path.join(base_dir, "data"), train = False, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
testloader = DataLoader(testset, batch_size = batch_size, shuffle = False)


# ====================== #
# Model
# ====================== #
model = SimpleFCModel(fc1_scale = None).to(device)
criterion = nn.CrossEntropyLoss()


print("Stage 1: QAT pretrain with dynamic FC1 activation scale")
optimizer = optim.Adam(model.parameters(), lr = pretrain_lr)
for epoch in range(pretrain_epochs):
    average_loss = train_one_epoch(trainloader, model, criterion, optimizer, device, epoch, pretrain_epochs)
    test_acc = check_accuracy(testloader, model, device)
    print("=============================")
    print(f"Pretrain Epoch [{epoch + 1}/{pretrain_epochs}], Average Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=============================\n")


print("Stage 2: calibrate fixed FC1 activation scale")
fc1_scale, fc1_clip_value = calibrate_fc1_scale(
    trainloader,
    model,
    device,
    percentile = calibration_percentile,
)
model.set_fc1_scale(fc1_scale)
print(f"FC1 calibration percentile: {calibration_percentile}")
print(f"FC1 clip value: {fc1_clip_value:.8f}")
print(f"FC1 output scale: {fc1_scale:.12f}")


print("Stage 3: fine tune with fixed FC1 activation scale")
optimizer = optim.Adam(model.parameters(), lr = finetune_lr)
for epoch in range(finetune_epochs):
    average_loss = train_one_epoch(trainloader, model, criterion, optimizer, device, epoch, finetune_epochs)
    test_acc = check_accuracy(testloader, model, device)
    print("=============================")
    print(f"Finetune Epoch [{epoch + 1}/{finetune_epochs}], Average Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=============================\n")


print("Saving hardware-facing weights and scales")
scales = save_hardware_artifacts(model, directory = os.path.join(base_dir, "weights"))
print("Saved:")
print("  experiment4_FIST/weights/fc1_int4.pth        int4 tensor, shape [118, 484]")
print("  experiment4_FIST/weights/fc2_int4.pth        int4 tensor, shape [10, 118]")
print("  experiment4_FIST/weights/fc1.pth  float tensor for analysis/resume")
print("  experiment4_FIST/weights/fc2.pth  float tensor for analysis/resume")
print("  experiment4_FIST/weights/scales.json")
print(scales)
