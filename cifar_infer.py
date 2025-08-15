# cifar_infer.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cifar_model import CifarCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
loader = DataLoader(test_ds, batch_size=1, shuffle=True)

model = CifarCNN().to(device)
model.load_state_dict(torch.load("cifar_cnn.pt", map_location=device))
model.eval()

with torch.no_grad():
    correct = 0
    for i, (img, label) in enumerate(loader):
        img, label = img.to(device), label.to(device)
        pred = model(img).argmax(1).item()
        print(f"[{i+1}] 預測: {classes[pred]:<10}  實際: {classes[label.item()]}")
        correct += int(pred == label.item())
        if i >= 9:  # 看前 10 筆
            break
    print(f"前 10 筆正確數：{correct}/10")