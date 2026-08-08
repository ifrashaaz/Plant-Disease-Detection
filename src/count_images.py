import os

dataset_path = "dataset/PlantVillage/PlantVillage"

print("Image count for each class:\n")

total_images = 0

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        image_count = len(images)

        print(class_name, ":", image_count)

        total_images += image_count

print("\nTotal images:", total_images)