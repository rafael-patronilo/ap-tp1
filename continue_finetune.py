import finetune
import sys
import torch
import os

filepath = sys.argv[1]
from_epoch = int(sys.argv[2])
epochs = int(sys.argv[3])

model_name = None

parts = os.path.split(filepath)[1].split("_")

if parts[1] == "best":
    model_name = "_".join(parts[2:-1])
else:
    model_name = "_".join(parts[1:-1])

model = torch.load(filepath, map_location=finetune.device)
finetune.train_fine_tuning(model_name, model, 0.001, 
                           param_group=True, from_epoch=from_epoch, epochs=epochs)