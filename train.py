import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from loss import CrossEntropyLabelSmooth
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import argparse


class FoodDataset(Dataset):
    def __init__(self, file, transform=None, mode="train"):
        self.transforms = transform
        self.mode = mode
        with open(file, "r") as f:
            self.image_list = f.readlines()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        label = None
        if self.mode == "train":
            image, label = self.image_list[index].split("\n")[0].split("\t")
            label = int(label)
        else:
            image = self.image_list[index].split("\n")[0]
        image = Image.open(image).convert("RGB")
        image = self.transforms(image)
        if self.mode == "train":
            return image, label
        else:
            return image


def evaluate(prediction, ground_truth):
    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
    return num_correct / len(prediction)


def get_model(model_name, num_classes):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def main(args):
    model_name = args.model_name
    num_classes = 6
    train_model = get_model(model_name, num_classes)

    transforms_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Pad(10, 10),
            transforms.RandomRotation(45),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transforms_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # train_ds = FoodDataset("./data/train.txt", transform=transforms_train)
    # val_ds = FoodDataset("./data/val.txt", transform=transforms_test)
    # test_ds = FoodDataset("./data/test.txt", transform=transforms_test)
    train_ds = FoodDataset("./data/train-6.txt", transform=transforms_train)
    val_ds = FoodDataset("./data/val-6.txt", transform=transforms_test)
    test_ds = FoodDataset("./data/test-6.txt", transform=transforms_test)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=8)

    output_dir = "checkpoint/{}-6".format(model_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = dict()
    results["train_loss"] = list()
    results["train_acc"] = list()
    results["val_loss"] = list()
    results["val_acc"] = list()

    cuda_num = 2
    device = "cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu"
    ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes, device=device)
    # ce_loss = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    if model_name.startswith("resnet"):
        for param in train_model.parameters():
            param.requires_grad = False
        for param in train_model.fc.parameters():
            param.requires_grad = True
        for i in range(5):
            train_model.train()
            train_model.to(device)
            for img, label in tqdm(train_dl):
                img = img.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = train_model(img)
                loss = ce_loss(output, label)
                loss.backward()
                optimizer.step()
        for param in train_model.parameters():
            param.requires_grad = True

    epoch = 100
    highest_acc = {"epoch": 0, "accuracy": 0}
    for ep in range(epoch):
        train_model.train()
        train_model.to(device)
        count = 0
        running_loss = 0.0
        validation_loss = 0.0
        output_list = []
        ground_truth_list = []
        for img, label in tqdm(train_dl):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = train_model(img)
            loss = ce_loss(output, label)
            count += 1
            prediction = torch.argmax(output, dim=1)
            output_list.extend(prediction.detach().cpu())
            ground_truth_list.extend(label.cpu())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        accuracy = evaluate(output_list, ground_truth_list)
        print(
            f"Epoch[{ep}] training accuracy: {accuracy} "
            f'training loss: {running_loss / count:.3e} Base Lr: {optimizer.param_groups[0]["lr"]:.5e}'
        )

        if ep % 10 == 0:
            torch.save(
                train_model.state_dict(),
                output_dir + "/{}_".format(model_name) + str(ep) + ".pth",
            )
            results["train_loss"].append(running_loss / count)
            results["train_acc"].append(accuracy)

            train_model.eval()
            count = 0
            output_list = []
            ground_truth_list = []
            for img, label in tqdm(val_dl):
                with torch.no_grad():
                    img = img.to(device)
                    lbl = label.to(device)
                    output = train_model(img)
                    val_loss = ce_loss(output, lbl)
                    validation_loss += val_loss.item()
                    count += 1
                    prediction = torch.argmax(output, dim=1)
                    output_list.extend(prediction.detach().cpu())
                    ground_truth_list.extend(label)
            accuracy = evaluate(output_list, ground_truth_list)
            if accuracy > highest_acc["accuracy"]:
                highest_acc["accuracy"] = accuracy
                highest_acc["epoch"] = ep
            print(f"Accuracy: {accuracy}    Epoch:{ep}")

            results["val_loss"].append(validation_loss / count)
            results["val_acc"].append(accuracy)

    torch.save(
        train_model.state_dict(),
        output_dir + "/{}_".format(model_name) + "final" + ".pth",
    )
    print(
        "highest_acc: {}  epoch: {}".format(
            highest_acc["accuracy"], highest_acc["epoch"]
        )
    )

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv("{}/results.csv".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="vgg16", help="name of CNN model"
    )
    args = parser.parse_args()
    print(args)
    main(args)
