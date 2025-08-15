# cifar_train.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cifar_model import CifarCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 超參數
epochs = 30
batch_size = 128
lr = 0.01
weight_decay = 5e-4

# CIFAR-10 的標準化（官方常用均值/方差）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# 資料前處理：訓練集做簡單增強，測試集只正規化
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.1),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# 資料集與載入器（第一次會自動下載）
train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

pin = device.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

# 建模、損失、優化器
model = CifarCNN().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 混合精度（有 GPU 時啟用可加速）
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

for epoch in range(1, epochs + 1):
    # ---------- Train ----------
    model.train()
    total = correct = 0
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---------- Eval ----------
    model.eval()
    t_total = t_correct = 0
    t_loss_sum = 0.0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            t_loss_sum += loss.item() * images.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total += labels.size(0)
    test_loss = t_loss_sum / t_total
    test_acc = t_correct / t_total

    print(f"Epoch {epoch}: train loss={train_loss:.4f}, acc={train_acc:.4f} | test loss={test_loss:.4f}, acc={test_acc:.4f}")
    scheduler.step()

# 存模型權重
torch.save(model.state_dict(), "cifar_cnn.pt")
print("Saved weights to cifar_cnn.pt")
