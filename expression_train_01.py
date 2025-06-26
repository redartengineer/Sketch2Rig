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
    
#Creates a transform object that, when applied to an image, will first resize the image to __x__pixels 
#and then convert it into a PyTorch tensor with scaled pixel values.
#Preparing image data.
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

dataset = ExpressionDataset("labels_01.csv", "images", transform)
dataloader = DataLoader (dataset, batch_size=128, shuffle=True)

#Defines CNN Model.
class ExpressionNet (nn.Module):
    def __init__(self):
        super(ExpressionNet, self).__init__()
        self.conv = nn.Sequential (
            nn.Conv2d(1, 16, 3, padding = 1),        #nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential (
            nn.Flatten(),
            nn.Linear(32*16*16, 128),               #nn.linear(batch_size, channels, height, width)
            nn.ReLU(),
            nn.Dropout(0.3),                        #Prevents Overfitting
            nn.Linear(128,18),
        )

#Forward Pass - Passes data through network.
    def forward (self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
#Allows Model Training to be done in GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExpressionNet().to(device)
    
#Optimization - Uses Adam Optimizer in PyTorch.  
#Used to update the parameters of the neural network model during training.
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

#Model Training loop
num_epoch = 100
for epoch in range (num_epoch):
    running_loss = 0.0
    for imgs, labels in dataloader:  # Loop over batches
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
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "expression_model_01.pth")
print("✅ Model saved.")