# cifar_model.py
import torch.nn as nn

class CifarCNN(nn.Module):
    """
    CIFAR-10 (RGB 3×32×32, 10 classes)
    Conv→Conv→Pool→Conv→Conv→Pool→FC→FC
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 32→16

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # 64*8*8 = 4096
            nn.Linear(64*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
