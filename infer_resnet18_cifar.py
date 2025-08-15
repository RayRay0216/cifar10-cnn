# infer_resnet18_cifar.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()
test_tf = transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224), preprocess])

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
loader = DataLoader(test_ds, batch_size=1, shuffle=True)

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
state = torch.load("resnet18_cifar10.pt", map_location=device)
model.load_state_dict(state)
model = model.to(device).eval()

with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
    ok = 0
    for i, (img, label) in enumerate(loader):
        img, label = img.to(device), label.to(device)
        pred = model(img).argmax(1).item()
        print(f"[{i+1}] 預測: {classes[pred]:<10}  實際: {classes[label.item()]}")
        ok += int(pred == label.item())
        if i >= 9: break
    print(f"前 10 筆正確數：{ok}/10")
