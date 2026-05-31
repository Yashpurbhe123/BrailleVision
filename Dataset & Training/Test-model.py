# ─────────────────────────────────────────────
# BRAILLE IMAGE UPLOAD + PREDICTION
# Opens local file picker automatically
# Works in VS Code Python Script
# ─────────────────────────────────────────────

import torch
import json
import timm
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────[]

SAVE_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# LOAD CLASS MAP
# ─────────────────────────────────────────────

with open(f"{SAVE_DIR}/class_map.json") as f:
    data = json.load(f)

idx_to_char = {
    int(k): v
    for k, v in data["idx_to_char"].items()
}

NUM_CLASSES = len(idx_to_char)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

ckpt = torch.load(
    f"{SAVE_DIR}/best_model.pth",
    map_location=DEVICE
)

model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(DEVICE)

print("✅ Model Loaded")
print(f"📊 Validation Accuracy: {ckpt['val_acc'] * 100:.2f}%")

# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ─────────────────────────────────────────────
# OPEN FILE PICKER
# ─────────────────────────────────────────────

print("\n📂 Select a Braille image...")

root = Tk()
root.withdraw()

file_path = askopenfilename(
    title="Select Braille Image",
    filetypes=[
        ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
        ("All Files", "*.*")
    ]
)

# ─────────────────────────────────────────────
# CHECK FILE
# ─────────────────────────────────────────────

if not file_path:
    print("❌ No image selected")
    exit()

print(f"\n✅ Selected Image:\n{file_path}")

# ─────────────────────────────────────────────
# LOAD IMAGE
# ─────────────────────────────────────────────

img = Image.open(file_path).convert("RGB")

# ─────────────────────────────────────────────
# MODEL PREDICTION
# ─────────────────────────────────────────────

x = tfm(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]

top5_v, top5_i = probs.topk(5)

pred_char = idx_to_char[top5_i[0].item()]
pred_conf = top5_v[0].item() * 100

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Input image
ax1.imshow(img)
ax1.set_title("Uploaded Braille Image")
ax1.axis("off")

# Top predictions
labels = [idx_to_char[i.item()] for i in top5_i]
values = [v.item() * 100 for v in top5_v]

colors = [
    "#2ecc71" if i == 0 else "#3498db"
    for i in range(5)
]

bars = ax2.barh(
    labels[::-1],
    values[::-1],
    color=colors[::-1]
)

ax2.set_xlabel("Confidence (%)")
ax2.set_title("Top 5 Predictions")

for bar, val in zip(bars, values[::-1]):
    ax2.text(
        bar.get_width() + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center"
    )

ax2.set_xlim(0, 110)

plt.suptitle(
    f'Prediction: "{pred_char}" | Confidence: {pred_conf:.1f}%',
    fontsize=15,
    fontweight='bold',
    color='green' if pred_conf > 70 else 'orange'
)

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# CONSOLE OUTPUT
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print(f"🔤 Predicted Character : {pred_char}")
print(f"📈 Confidence          : {pred_conf:.1f}%")

if pred_conf > 70:
    print("✅ Status              : Confident Prediction")
else:
    print("⚠️ Status              : Low Confidence")

print("=" * 50)