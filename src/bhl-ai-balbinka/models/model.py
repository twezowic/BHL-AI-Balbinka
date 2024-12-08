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

        images_dir = os.path.join(self.data_path)
        labels_file = os.path.join(self.data_path, 'meta_data.csv')
        label_data = pd.read_csv(labels_file)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensors = []
        for interesting_files in main_load(self.data_path):
            for file in interesting_files:
                image = Image.open(file).convert('L')
                image_tensor = transform(image)
                image_tensors.append(image_tensor)
            folder_tensor = torch.stack(image_tensors)
            id = '_'.join(file.split(os.sep)[-3:-1])
            labels.append(normalize(label_data, id))
            image_data.append(folder_tensor)


            # path = os.path.join(images_dir, folder)

            # if os.path.isdir(path):
            #     image_tensors = []
            #     for file_name in os.listdir(path):
            #         file = os.path.join(path, file_name)
            #         image = Image.open(file).convert('L')
            #         #image = Image.open(file).convert('RGB')
            #         image_tensor = transform(image)
            #         image_tensors.append(image_tensor)
            #     folder_tensor = torch.stack(image_tensors)  # Stack images into one tensor for this folder
            #     id = '_'.join(folder.split('\\')[-2:])
            #     labels.append(normalize(label_data, id))
            #     image_data.append(folder_tensor)


        return image_data, labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images = self.data[idx]  # Assuming list of file paths
        # image_stack = []
        # for channel, img_path in zip(self.channels, images):
            # img = Image.open(img_path)
            # if self.transform:
            #     img = self.transform(img)
            # image_stack.append(img)
        #images_tensor = torch.stack(image_stack)  # Shape: (channels, H, W)
        label = self.labels[idx]
        return images, torch.tensor(label, dtype=torch.float32)

# Define the model
class SolarFlareModel(nn.Module):
    def __init__(self):
        super(SolarFlareModel, self).__init__()

        self.my_test = nn.Sequential(
            nn.Conv2d(40, 128, kernel_size=3, padding=1, stride=2), # 256x256 => 128x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=2),  # 128x128 => 64x64
            nn.ReLU()
        )
        # self.time_distributed = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )

        self.separable_blocks = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(2),
            nn.ReLU(),
            # nn.Conv2d(129, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(scales[3]),
            # nn.ReLU(),
            # nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(scales[4]),
            # nn.ReLU()
        )

        self.global_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(131072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Softmax(dim=1)
        )

    def forward(self, x, date_input):
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(-1, time_steps, H, W)
        x = self.my_test(x)
        batch_size, time_steps, H, W = x.size()
        x = x.view(1, 256, -1)
        # x = self.separable_blocks(x)
        # x = self.global_pool(x)
        x = self.separable_blocks(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# Training setup
base_path = os.path.join(get_data_dir(), 'SDOBenchmark-data-example')

params = {'dim': (4, 256, 256, 4),
          'batch_size': 1,
          'channels': ['magnetogram', '304', '131', '1700'],
          'shuffle': True}

train_dataset = SDOBenchmarkDataset(os.path.join(base_path, 'training'), params)
# train_dataset , validation_dataset = random_split(full_training_dataset, [0.9, 0.1])
test_dataset = SDOBenchmarkDataset(os.path.join(base_path, 'test'), params)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=params['shuffle'])
# val_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)
# val_loader_2 = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

model = SolarFlareModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        print(f"Validation Loss: {val_loss/test_amount:.4f}")

# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

