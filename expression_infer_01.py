import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import csv

#Defines the model
class ExpressionNet(nn.Module):
    def __init__(self):
        super(ExpressionNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 18)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

#Slider names in correct order
slider_names = [
    "eyelid_top", "eyelid_top_rotate", "eyelid_bottom", "eyelid_bottom_rotate",
    "iris_size", "pupil_size", "eye_look_x", "eye_look_y",
    "eyebrow_center_y", "eyebrow_center_x", "eyebrow_scale", "eyebrow_rotate",
    "eyebrow1", "eyebrow2", "eyebrow30", "eyebrow4", "eyebrow5", "eyebrow6"
]

#Load model
model = ExpressionNet()
model.load_state_dict(torch.load("expression_model_01.pth", map_location=torch.device('cpu')))
model.eval()

#Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

#Inference folder
inference_folder = "inference"
image_files = [f for f in os.listdir(inference_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("❌ No images found in 'inference/' folder.")
    exit()

#Create CSV output file
with open("inferred_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename"] + slider_names)  # Header

    for image_file in image_files:
        image_path = os.path.join(inference_folder, image_file)
        try:
            image = Image.open(image_path).convert("L")
            image = transform(image).unsqueeze(0)
        except Exception as e:
            print(f"❌ Error loading {image_file}: {e}")
            continue

        with torch.no_grad():
            output = model(image)
            values = output.squeeze().tolist()

        #Save to CSV
        writer.writerow([image_file] + values)

        #Print and prepare Houdini insertion
        print(f"\n🎯 Predicted sliders for '{image_file}':")
        predicted = []
        for name, val in zip(slider_names, values):
            print(f"  {name}: {val:.4f}")
            predicted.append(val)

        #Houdini-ready format:
        print("\n💡 Copy for Houdini:")
        print("predicted = [")
        for name, val in zip(slider_names, predicted):
            print(f"    {val:.4f},   # {name}")
        print("]")

