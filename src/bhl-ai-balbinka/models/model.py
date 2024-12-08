import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.read_csv import normalize

from data.load_data import main_load, get_data_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom dataset
class SDOBenchmarkDataset(Dataset):
    def __init__(self, data_path, params):
        self.data_path = data_path
        self.dim = params['dim']
        self.channels = params['channels']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * len(self.channels), std=[0.5] * len(self.channels))
        ])
        self.labels: pd.DataFrame = None
        self.data, self.labels = self.load_data()


    def load_data(self):
        # Prepare paths and labels manually
        image_data = []
        labels = []

        labels_file = os.path.join(self.data_path, 'meta_data.csv')
        label_data = pd.read_csv(labels_file)
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5] * len(self.channels), std=[0.5] * len(self.channels))
        ])
        for interesting_files in main_load(self.data_path):
            image_tensors = []
            for file in interesting_files:
                image = Image.open(file).convert('L')
                image_tensor = transform(image)
                image_tensors.append(image_tensor)
            folder_tensor = torch.stack(image_tensors)
            id = '_'.join(file.split(os.sep)[-3:-1])
            labels.append(normalize(label_data, id))
            image_data.append(folder_tensor)
        return image_data, labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]
        label = self.labels[idx]
        return images, torch.tensor(label, dtype=torch.float32)

# Define the model
class SolarFlareModel(nn.Module):
    def __init__(self):
        super(SolarFlareModel, self).__init__()

        self.my_test = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1, stride=2), # 256x256 => 128x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64x64
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.separable_blocks = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # self.global_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )
        self.fv = nn.Linear(32, 1)

    def forward(self, x, date_input):
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(-1, time_steps, H, W)

        x = self.my_test(x)
        batch_size, time_steps, H, W = x.size()
        x = x.view(8, 32, -1)
        x = self.separable_blocks(x)
        x = x.view(4, -1)
        x = self.fc(x)
        x  = x.view(-1)
        return self.fv(x)

base_path = os.path.join(get_data_dir(), 'SDOBenchmark-data-example')

params = {'dim': (4, 256, 256, 4),
          'batch_size': 1,
          'channels': ['magnetogram'],
          'shuffle': True}

train_dataset = SDOBenchmarkDataset(os.path.join(base_path, 'training'), params)
train_dataset , _ = random_split(train_dataset, [0.5, 0.5])
test_dataset = SDOBenchmarkDataset(os.path.join(base_path, 'test'), params)

print("train amount:", len(train_dataset))
print("test amount:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=params['shuffle'])
# val_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)
# val_loader_2 = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

model = SolarFlareModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPSILON = 1/100
# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        target = 0
        model.eval()
        val_loss = 0.0
        test_amount = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, labels)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                test_amount += 1

                if (abs(target - outputs) < abs(np.log(EPSILON))):
                    target += 1

        print(f"Validation Loss: {val_loss/test_amount:.4f}")
        print(target/test_amount * 100)
# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

torch.save(model.state_dict(), 'balbinka_model.pth')