#Update 02

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os

#Locate labeled CSV file and images.  
class ExpressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert("L")
        label = self.labels.iloc[idx, 1:].astype(float).values
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
    
#Preparing image data. Enhance model robustness and prevent overfitting.
#Simulate real-world drawing inconsistencies and add beneficial randomness to the training process.
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = ExpressionDataset("train_labels_01.csv", "images", train_transform)
val_dataset = ExpressionDataset("val_labels_01.csv", "images", val_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

#Defines CNN Model.
class ExpressionNetV2 (nn.Module):
    def __init__(self):
        super(ExpressionNetV2, self).__init__()
        self.conv = nn.Sequential (
            nn.Conv2d(1, 32, 5, padding = 1),        #nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc = nn.Sequential (
            nn.Flatten(),
            nn.Linear(256*7*7, 512),               #nn.linear(batch_size, channels, height, width)
            nn.ReLU(),
            nn.Dropout(0.2),                        #Prevents Overfitting
            nn.Linear(512,18),
        )

#Forward Pass - Passes data through network.
    def forward (self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

#Allows Model Training to be done in GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExpressionNetV2().to(device)

#Optimization - Uses Adam Optimizer in PyTorch.  
#Used to update the parameters of the neural network model during training.
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

#Training loop with validation
#Model Training loop
num_epoch = 800
for epoch in range (num_epoch):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:  # Loop over batches
        # Move data to the appropriate device (CPU or GPU)
        imgs, labels = imgs.to(device), labels.to(device)
        #Forwards Pass to get output
        outputs = model(imgs)
        #Calculates loss
        loss = criterion(outputs, labels)
        #Backwards Pass
        optimizer.zero_grad()       #Clear gradients parameters
        loss.backward()             #Getting gradients parameter
        #Update Weights
        optimizer.step()            #Updating parameters
        #Print Progress
        running_loss += loss.item()

 #Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")


torch.save(model.state_dict(), "expression_model_01.pth")
print("✅ Model saved.")