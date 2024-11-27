import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy

class CIFAR10Classifier(pl.LightningModule):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        # Define a simple CNN architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes in CIFAR-10
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = Accuracy(num_classes=10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        val_loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.accuracy(outputs, labels)
        return {"val_loss": val_loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {"val_loss": avg_loss, "val_acc": avg_acc}

    def configure_optimizers(self):
        # Use Adam optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize CIFAR-10 images
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize the model
    model = CIFAR10Classifier()

    # Setup Trainer
    trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)

    # Train the model
    trainer.fit(model, train_loader, val_loader)