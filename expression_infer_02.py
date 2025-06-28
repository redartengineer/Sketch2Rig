#Update 02

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import csv

# Defines the CNN model
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

# Slider names
slider_names = [
    "eyelid_top", "eyelid_top_rotate", "eyelid_bottom", "eyelid_bottom_rotate",
    "iris_size", "pupil_size", "eye_look_x", "eye_look_y",
    "eyebrow_center_y", "eyebrow_center_x", "eyebrow_scale", "eyebrow_rotate",
    "eyebrow1", "eyebrow2", "eyebrow30", "eyebrow4", "eyebrow5", "eyebrow6"
]

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Slider min and max values.
slider_min = torch.tensor([
    0.0, -35.0, 0.0, -35.0,
    0.5, 0.3, -0.03, -1.0,
    -0.05, -0.5, 0.5, -35.0,
    -0.3, -0.3, -0.3, -0.3, -0.3, -0.3
], dtype=torch.float32).to(device)

slider_max = torch.tensor([
    1.0, 35.0, 1.0, 35.0,
    1.5, 1.2, 1.0, 1.0,
    0.5, 0.5, 0.5, 60.0,
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3
], dtype=torch.float32).to(device)

# Load and prepare model
model = ExpressionNetV2().to(device)
model.load_state_dict(torch.load("expression_model_01.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load images from folder
inference_folder = "inference"
image_files = [f for f in os.listdir(inference_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("❌ No images found in 'inference/' folder.")
    exit()

# Save predictions to CSV
with open("inferred_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename"] + slider_names)

    for image_file in image_files:
        image_path = os.path.join(inference_folder, image_file)
        try:
            image = Image.open(image_path).convert("L")
            image = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"❌ Error loading {image_file}: {e}")
            continue

        with torch.no_grad():
            output = model(image)
            output = torch.max(torch.min(output, slider_max), slider_min)
            values = output.squeeze().tolist()

        # Save values to CSV
        writer.writerow([image_file] + values)

        # Print for Houdini
        print(f"\n🎯 Predicted sliders for '{image_file}':")
        for name, val in zip(slider_names, values):
            print(f"  {name}: {val:.4f}")

        print("\n💡 Copy for Houdini:")
        print("predicted = [")
        for name, val in zip(slider_names, values):
            print(f"    {val:.4f},   # {name}")
        print("]")
