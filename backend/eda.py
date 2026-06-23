import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# =====================================
# Dataset Path
# =====================================

DATASET_PATH = r"Dataset/PlantVillage"

OUTPUT_DIR = r"outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# Load Classes
# =====================================

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print(f"\nTotal Classes: {len(classes)}")

# =====================================
# Class Distribution
# =====================================

class_counts = {}

for cls in classes:

    class_path = os.path.join(DATASET_PATH, cls)

    count = len([
        file for file in os.listdir(class_path)
        if file.lower().endswith(
            ('.jpg', '.jpeg', '.png')
        )
    ])

    class_counts[cls] = count

df = pd.DataFrame(
    class_counts.items(),
    columns=["Class", "Images"]
)

print(df)

# Save CSV

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution.csv"
    ),
    index=False
)

# =====================================
# Plot Class Distribution
# =====================================

plt.figure(figsize=(14, 6))

sns.barplot(
    data=df,
    x="Class",
    y="Images"
)

plt.xticks(rotation=90)
plt.title("Class Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution.png"
    )
)

plt.show()

# =====================================
# Sample Images
# =====================================
import math

num_classes = len(classes)
cols = 5
rows = math.ceil(num_classes / cols)

plt.figure(figsize=(15, rows * 3))

for i, cls in enumerate(classes):

    class_path = os.path.join(DATASET_PATH, cls)

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        continue

    sample = random.choice(images)

    img_path = os.path.join(class_path, sample)

    image = Image.open(img_path)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(image)
    plt.title(cls[:20])
    plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================
# Image Dimension Analysis
# =====================================

widths = []
heights = []

for cls in classes:

    class_path = os.path.join(DATASET_PATH, cls)

    for img_file in os.listdir(class_path):

        if img_file.lower().endswith(
            ('.jpg', '.jpeg', '.png')
        ):

            img_path = os.path.join(
                class_path,
                img_file
            )

            img = Image.open(img_path)

            widths.append(img.size[0])
            heights.append(img.size[1])

print("\nImage Statistics")
print("----------------")
print("Minimum Width :", min(widths))
print("Maximum Width :", max(widths))
print("Minimum Height:", min(heights))
print("Maximum Height:", max(heights))
