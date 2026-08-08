from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Location of our dataset
DATASET_DIR = Path("dataset/PlantVillage")

# Location where we will save the split files
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Image file types we will use
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Store image paths and class names
data = []

# Go through every class folder
for class_folder in sorted(DATASET_DIR.iterdir()):

    if class_folder.is_dir():

        class_name = class_folder.name

        for image_file in class_folder.iterdir():

            if image_file.suffix.lower() in IMAGE_EXTENSIONS:

                data.append({
                    "image_path": image_file.as_posix(),
                    "label": class_name
                })

# Convert the information into a DataFrame
df = pd.DataFrame(data)

print("Total images:", len(df))
print("Total classes:", df["label"].nunique())

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split: divide the temporary 30% into
# 15% validation and 15% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Add the split name
train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# Combine everything
final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# Save the CSV files
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Save one combined file too
final_df.to_csv(OUTPUT_DIR / "dataset_split.csv", index=False)

print("\nDataset split completed!")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))