import pandas as pd
import numpy as np
import os
from torchvision.io import read_image


img_labels = pd.read_csv("./data/images/images/train.csv")
images = []
for i, row in img_labels.iterrows():
  name = row["image_id"]
  label = row["main_type"]
  print(i, name, label)
  img_path = os.path.join(
            "./data/images/images/train", name.replace(";", "") + ".png"
        )
  images.append(read_image(img_path).float().numpy())
  if i > 4000:
      break
print("Writing back")
with open('./data/images/images/train.npy', 'wb') as file:
    np.save(file, np.concatenate(images))