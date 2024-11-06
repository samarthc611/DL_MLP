import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MlpBlock(nn.Module):
    def __init__(self, mlp_dim):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(mlp_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = nn.GELU()(y)
        return self.fc2(y)

class MixerBlock(nn.Module):

    def __init__(self, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=128)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=128)
        self.mlp_tokens = MlpBlock(tokens_mlp_dim)
        self.mlp_channels = MlpBlock(channels_mlp_dim)

    def forward(self, x):
        y = self.layer_norm1(x)
        y = y.transpose(1, 2)  # Swap axes
        y = self.mlp_tokens(y)
        y = y.transpose(1, 2)  # Swap axes back
        x = x + y
        y = self.layer_norm2(x)
        return x + self.mlp_channels(y)

class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MlpMixer, self).__init__()
        self.conv = nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.mixer_blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.conv(x)  # (N, C, H, W)
        x = x.flatten(2).transpose(1, 2)  # (N, H*W, C)

        for block in self.mixer_blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

def load_data(batch_size=32, img_size=(256, 256)):
    dataset_path = "Dataset/Public_Medical_Image_Datasets/covid19-pneumonia-dataset"

    train_dir = os.path.join(dataset_path, "train_dir")
    valid_dir = os.path.join(dataset_path, "valid_dir")
    test_dir = os.path.join(dataset_path, "test_dir")

    if not all(os.path.exists(d) for d in [train_dir, valid_dir, test_dir]):
        raise FileNotFoundError(f"One or more dataset directories not found. {d}")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    valid_ds = datasets.ImageFolder(valid_dir, transform=transform)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def train_step(model, optimizer, criterion, batch):
    model.train()
    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return loss.item(), accuracy.item()

def train_model(model, train_loader, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for batch in train_loader:
            loss, accuracy = train_step(model, optimizer, criterion, batch)
            total_loss += loss
            total_accuracy += accuracy

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {total_accuracy / len(train_loader) * 100:.2f}%")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
num_classes = 3
model = MlpMixer(num_classes=num_classes, num_blocks=8, patch_size=4, hidden_dim=128, tokens_mlp_dim=256, channels_mlp_dim=512).to(device)

# Load data
train_loader, valid_loader, test_loader = load_data()

# Train the model
train_model(model, train_loader, num_epochs=10)

