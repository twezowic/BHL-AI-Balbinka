import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.read_csv import find_peakflux

from data.load_data import main_load, get_data_dir

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the custom dataset
class SDOBenchmarkDataset(Dataset):
    def __init__(self, data_path, params):
        self.data_path = data_path
        self.dim = params['dim']
        self.channels = params['channels']
        self.label_func = params['label_func']
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
        # labels = [1.88E-06, 7.53E-07, 3.06E-06, 5.76E-06, 1.00E-09] #@TODO TEMP

        # for _, row in label_data.iterrows():
        #     image_id = row['image_id']
        #     label = row['label']

        #     # Collect all channel paths for this sample
        #     channel_paths = [
        #         os.path.join(images_dir, f"{image_id}_{channel}.png") for channel in self.channels
        #     ]
        #     if all(os.path.exists(p) for p in channel_paths):  # Check all channels exist
        #         image_data.append(channel_paths)
        #         labels.append(label)
        transform = transforms.Compose([
            transforms.ToTensor(),
        # Optionally, you can normalize the images if needed:
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        for folder in main_load(self.data_path):
            path = os.path.join(images_dir, folder)
            if os.path.isdir(path):
                image_tensors = []
                for file_name in os.listdir(path):
                    file = os.path.join(path, file_name)
                    # image = Image.open(file).convert('L')
                    image = Image.open(file).convert('RGB')
                    image_tensor = transform(image)
                    image_tensors.append(image_tensor)
                folder_tensor = torch.stack(image_tensors)  # Stack images into one tensor for this folder
                id = '_'.join(folder.split('\\')[-2:])
                labels.append(find_peakflux(label_data, id))
                image_data.append(folder_tensor)


        return image_data, label_data


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
        label = self.labels["peak_flux"][idx]
        return images, torch.tensor(label, dtype=torch.float64)

# Define the model
class SolarFlareModel(nn.Module):
    def __init__(self, scales, num_categories):
        super(SolarFlareModel, self).__init__()

        self.time_distributed = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.separable_blocks = nn.Sequential(
            nn.Conv2d(64, scales[3], kernel_size=1, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(scales[3]),
            nn.ReLU(),
            nn.Conv2d(scales[3], scales[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(scales[3]),
            nn.ReLU(),
            nn.Conv2d(scales[3], scales[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(scales[4]),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(scales[4] + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_categories),
            nn.Softmax(dim=1)
        )

    def forward(self, images, date_input):
        batch_size, time_steps, C, H, W = images.size()
        x = images.view(-1, C, H, W)
        x = self.time_distributed(x)
        x = x.view(batch_size, time_steps, -1)
        x = self.separable_blocks(x)
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        x = torch.cat([x, date_input], dim=1)
        return self.fc(x)

# Training setup
base_path = os.path.join(get_data_dir(), 'SDOBenchmark-data-example')

params = {'dim': (4, 256, 256, 4),
          'batch_size': 16,
          'channels': ['magnetogram', '304', '131', '1700'],
          'shuffle': True,
          'label_func': lambda y: torch.tensor(np.floor(np.log10(y) + 9).astype(int))}

training_dataset = SDOBenchmarkDataset(os.path.join(base_path, 'training'), params)
validation_dataset = SDOBenchmarkDataset(os.path.join(base_path, 'test'), params)

train_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=params['shuffle'])
val_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)

scales = [64, 64 * 3, 1, 128, 256]
num_categories = 7

model = SolarFlareModel(scales, num_categories).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
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
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, labels)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
