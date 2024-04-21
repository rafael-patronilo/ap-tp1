import torch
import os
import pandas as pd
from PIL import Image
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


type_count = {
    "Flying": 0,
    "Fairy": 0,
    "Ice": 0,
    "Steel": 0,
    "Dragon": 0,
    "Ghost": 0,
    "Fighting": 0,
    "Dark": 0,
    "Ground": 0,
    "Poison": 0,
    "Electric": 0,
    "Fire": 0,
    "Rock": 0,
    "Psychic": 0,
    "Bug": 0,
    "Grass": 0,
    "Water": 0,
    "Normal": 0,
}

output_path_csv = "./data/images/image_aug/train.csv"
output_path_images = "./data/images/image_aug/images/"
input_path = "./data/images/images/train/"
df = pd.read_csv("./data/images/images/train.csv")
header = ["image_id", "main_type", "secondary_type"]
with open(output_path_csv, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
print(os.path.exists(output_path_images))
for index, row in df.iterrows():
    image_name = row["image_id"]
    try:
        with Image.open(os.path.join(input_path, image_name + ".png")) as img:
            if type_count[row["main_type"]] <= 600:
                type_count[row["main_type"]] += 1
                img.save(output_path_images + f"{image_name}.png")
                data_to_append = {
                    "image_id": [f"{image_name}"],
                    "main_type": [row["main_type"]],
                    "secondary_type": [row["secondary_type"]],
                }
                new_data = pd.DataFrame(data_to_append)
                new_data.to_csv(output_path_csv, mode="a", header=False, index=False)
            if type_count[row["main_type"]] <= 600:
                type_count[row["main_type"]] += 1
                rotated = img.rotate(180)
                rotated.save(output_path_images + f"{image_name}_rotated.png")
                data_to_append = {
                    "image_id": [f"{image_name}_rotated"],
                    "main_type": [row["main_type"]],
                    "secondary_type": [row["secondary_type"]],
                }
                new_data = pd.DataFrame(data_to_append)
                new_data.to_csv(output_path_csv, mode="a", header=False, index=False)
            if type_count[row["main_type"]] <= 600:
                type_count[row["main_type"]] += 1
                horizontal_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                horizontal_flipped.save(
                    output_path_images + f"{image_name}_horizontal_flipped.png"
                )
                data_to_append = {
                    "image_id": [f"{image_name}_horizontal_flipped"],
                    "main_type": [row["main_type"]],
                    "secondary_type": [row["secondary_type"]],
                }
                new_data = pd.DataFrame(data_to_append)
                new_data.to_csv(output_path_csv, mode="a", header=False, index=False)
            if type_count[row["main_type"]] <= 600:
                type_count[row["main_type"]] += 1
                vertical_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
                vertical_flipped.save(
                    output_path_images + f"{image_name}_vertical_flipped.png"
                )
                data_to_append = {
                    "image_id": [f"{image_name}_vertical_flipped"],
                    "main_type": [row["main_type"]],
                    "secondary_type": [row["secondary_type"]],
                }
                new_data = pd.DataFrame(data_to_append)
                new_data.to_csv(output_path_csv, mode="a", header=False, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
