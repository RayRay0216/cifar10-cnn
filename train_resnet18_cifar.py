# train_resnet18_cifar.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1) 預訓練權重與對應前處理（內含 Resize / Normalize）
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# 訓練用：加一點資料增強，再接上預處理
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    preprocess,
])
test_tf = transforms.Compose([
    transforms.Resize(232), transforms.CenterCrop(224),
    preprocess,
])

# 2) 資料集（自動下載）
train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

pin = device.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=pin)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=pin)

# 3) 模型：載入預訓練，改最後一層為 10 類
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 4) loss / optimizer / scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# 5) 混合精度（新 API）
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

def evaluate():
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum/total, correct/total

epochs = 30
for ep in range(1, epochs+1):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    tr_loss = loss_sum/total
    tr_acc  = correct/total
    te_loss, te_acc = evaluate()
    scheduler.step()
    print(f"Epoch {ep:02d}: train {tr_loss:.4f}/{tr_acc:.4f} | test {te_loss:.4f}/{te_acc:.4f}")

torch.save(model.state_dict(), "resnet18_cifar10.pt")
print("Saved weights to resnet18_cifar10.pt")
