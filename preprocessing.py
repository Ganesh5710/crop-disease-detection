import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================
# Dataset Path
# =====================================

DATASET_PATH = r"Dataset/PlantVillage"

OUTPUT_DIR = r"outputs/preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# Collect Images
# =====================================

image_paths = []
labels = []

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

for cls in classes:

    class_path = os.path.join(
        DATASET_PATH,
        cls
    )

    for img_file in os.listdir(class_path):

        if img_file.lower().endswith(
            ('.jpg', '.jpeg', '.png')
        ):

            image_paths.append(
                os.path.join(
                    class_path,
                    img_file
                )
            )

            labels.append(cls)

dataset_df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

print("Total Images:", len(dataset_df))

# =====================================
# Train / Validation / Test Split
# =====================================

train_df, temp_df = train_test_split(
    dataset_df,
    test_size=0.30,
    stratify=dataset_df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

print("\nDataset Split")
print("----------------")
print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))

# =====================================
# Save CSV Files
# =====================================

train_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "val.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test.csv"
    ),
    index=False
)

print("\nCSV files saved successfully.")
print("Images will be resized to 224x224 during model training.")