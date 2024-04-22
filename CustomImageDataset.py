# file was provided by teacher
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        target_transform=None,
        task="classification",
    ):
        self.img_labels = pd.read_csv(annotations_file)
        if "main_type" not in self.img_labels.columns:
            self.img_labels["main_type"] = "test"
        else:
            self.img_labels["main_type"] = self.img_labels["main_type"].str.lower()
            # self.img_labels['sub_type'] = self.img_labels['sub_type'].str.lower()
        classes = sorted(self.img_labels.iloc[:, 1].unique())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.task = task.lower()
        self.classes = [each_class.lower() for each_class in classes]
        self.class2idx = {self.classes[i].lower(): i for i in range(len(self.classes))}
        self.idx2class = {i: self.classes[i].lower() for i in range(len(self.classes))}
        self.per_class = [grouping for grouping in self.img_labels.groupby("main_type")]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.img_labels.iloc[idx, 0].replace(";", "") + ".png"
        )
        image = read_image(img_path).float()
        if self.img_labels.main_type.iloc[idx] == "test":
            label = -1
        if self.task == "classification":
            label = self.class2idx[self.img_labels.iloc[idx, 1].lower()]
        elif self.task == "multilabel_classification":
            label = self.class2idx[self.img_labels.iloc[idx, 1:-1].lower()]
        elif self.task == "semantic_segmentation":
            label = read_image(
                os.path.join(
                    self.img_dir,
                    self.img_labels.iloc[idx, 0]
                    + "mask"
                    + self.img_labels.iloc[idx, 0][-4:],
                )
            )
        else:
            raise ValueError(
                'Task must be either "classification", or "multilabel_classification", or "semantic_segmentation"'
            )

        if self.transform:
            image = self.transform(image)

        if label >= 0 and self.target_transform:
            label = self.target_transform(label)
        return image, label

    def show(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = read_image(img_path)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")
        plt.title(self.img_labels.iloc[idx, 1])
        plt.show()

    def show_mask(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.img_labels.iloc[idx, 0] + "mask" + self.img_labels.iloc[idx, 0][-4:],
        )
        image = read_image(img_path)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")
        plt.title(self.img_labels.iloc[idx, 1])
        plt.show()

    def show_mask_overlay(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = read_image(img_path)
        mask_path = os.path.join(
            self.img_dir,
            self.img_labels.iloc[idx, 0] + "mask" + self.img_labels.iloc[idx, 0][-4:],
        )
        mask = read_image(mask_path)
        plt.imshow(image.permute(1, 2, 0))
        plt.imshow(mask.permute(1, 2, 0), alpha=0.5)
        plt.axis("off")
        plt.title(self.img_labels.iloc[idx, 1])
        plt.show()

    def class_split(self, factor, random=np.random.default_rng()):
        def shuffle_and_select(group, slice_size):
            shuffled_group = group.sample(
                frac=1, random_state=random
            )  # Shuffle the group
            slice_a = shuffled_group.iloc[:slice_size]
            slice_b = shuffled_group.iloc[slice_size:]
            return slice_a.index, slice_b.index

        tuples = [shuffle_and_select(group, int(len(group) * factor)) for _name, group in self.per_class]
        random.shuffle(tuples)
        indices_a = np.concatenate([tup[0] for tup in tuples])
        indices_b = np.concatenate([tup[1] for tup in tuples])
        return Subset(self, indices_a), Subset(self, indices_b)

    def show_all(self):
        for idx in range(len(self.img_labels)):
            self.show(idx)
            # self.show_mask(idx)

    def show_all_overlay(self):
        for idx in range(len(self.img_labels)):
            self.show_mask_overlay(idx)

    def show_batch(self, images=None, labels=None, batch_size=4):
        grid_sz = int(np.sqrt(batch_size))
        fig, ax = plt.subplots(grid_sz, grid_sz, figsize=(10, 7))
        for i in range(min(batch_size, grid_sz * grid_sz)):
            if images == None:
                img_path = os.path.join(
                    self.img_dir, self.img_labels.iloc[i, 0] + ".png"
                )
                image = read_image(img_path)
            else:
                image = (images[i] - torch.min(images[i])) / (
                    torch.max(images[i]) - torch.min(images[i])
                )
            ax[i // grid_sz, i % grid_sz].imshow(image.permute(1, 2, 0))
            ax[i // grid_sz, i % grid_sz].axis("off")
            if self.task == "classification" and images == None:
                ax[i // grid_sz, i % grid_sz].set_title(self.img_labels.iloc[i, 1])
            else:
                ax[i // grid_sz, i % grid_sz].set_title(
                    str(labels[i].item()) + " : " + self.idx2class[labels[i].item()]
                )
                # ax[i//grid_sz,i%grid_sz].set_title(labels[i].item())
        plt.show()

    def get_class_distribution(self):
        class_dist = self.img_labels["main_type"].value_counts(ascending=True)
        return class_dist

    def get_class_weights(self):
        class_dist = self.get_class_distribution()
        return max(class_dist) / class_dist

    def get_class_idx(self, class_name):
        return self.class2idx[class_name.lower()]

    def get_class_string(self, idx):
        return self.idx2class[idx]

    def get_class_weights_tensor(self):
        class_weights = self.get_class_weights()
        # return tensor as float32
        return torch.tensor(
            [class_weights[self.idx2class[i]] for i in range(len(self.classes))],
            dtype=torch.float32,
        )

    def get_class_from_img_idx(self, idx):
        return self.img_labels.iloc[idx, 1]

    def get_class_string_from_numeric_label(self, idxs):
        if type(idxs) == int:
            return self.classes[idxs]
        else:
            out = []
            for idx in idxs:
                out.append(self.classes[idx])
            return out


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    # Test the CustomImageDataset class
    train_dataset = CustomImageDataset(
        annotations_file="./superimposed_images/train.csv",
        img_dir="./superimposed_images/train/",
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Load the test set
    test_dataset = CustomImageDataset(
        annotations_file="./superimposed_images/test.csv",
        img_dir="./superimposed_images/test/",
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
