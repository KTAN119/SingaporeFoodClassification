import os
import random

categories = [
    "Bak Chor Mee",
    "Char Kway Teow",
    "Chicken Rice",
    "Satay",
    "Oyster Omelette",
]

with open("data/train.txt", "w") as f:
    root_dir = "data/train"
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        for img in os.listdir(img_dir):
            f.write(root_dir + "/" + os.path.join(folder, img) + "\t" + str(i) + "\n")

with open("data/val.txt", "w") as f:
    root_dir = "data/val"
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        for img in os.listdir(img_dir):
            f.write(root_dir + "/" + os.path.join(folder, img) + "\t" + str(i) + "\n")

with open("data/test.txt", "w") as f:
    root_dir = "data/test"
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        for img in os.listdir(img_dir):
            f.write(root_dir + "/" + os.path.join(folder, img) + "\t" + str(i) + "\n")

with open("data/train-6.txt", "w") as f:
    root_dir = "data/train"
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        images = list(filter(lambda x: not x.startswith("."), os.listdir(img_dir)))
        for img in images:
            f.write(root_dir + "/" + os.path.join(folder, img) + "\t" + str(i) + "\n")

    other_categories = list(
        filter(
            lambda x: not x.startswith("."),
            list(set(os.listdir(root_dir)) - set(categories)),
        )
    )
    for category in other_categories:
        img_dir = os.path.join(root_dir, category)
        images = list(filter(lambda x: not x.startswith("."), os.listdir(img_dir)))
        sampled_images = random.sample(images, 15)
        for img in sampled_images:
            f.write(root_dir + "/" + os.path.join(category, img) + "\t" + str(5) + "\n")

with open("data/val-6.txt", "w") as f:
    root_dir = "data/val"
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        images = list(filter(lambda x: not x.startswith("."), os.listdir(img_dir)))
        for img in images:
            f.write(root_dir + "/" + os.path.join(folder, img) + "\t" + str(i) + "\n")

    other_categories = list(
        filter(
            lambda x: not x.startswith("."),
            list(set(os.listdir(root_dir)) - set(categories)),
        )
    )
    for category in other_categories:
        img_dir = os.path.join(root_dir, category)
        images = list(filter(lambda x: not x.startswith("."), os.listdir(img_dir)))
        sampled_images = random.sample(images, 4)
        for img in sampled_images:
            f.write(root_dir + "/" + os.path.join(category, img) + "\t" + str(5) + "\n")

with open("data/test-6.txt", "w") as f:
    root_dir = "data/test"
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        images = list(filter(lambda x: not x.startswith("."), os.listdir(img_dir)))
        for img in images:
            f.write(root_dir + "/" + os.path.join(folder, img) + "\t" + str(i) + "\n")

    other_categories = list(
        filter(
            lambda x: not x.startswith("."),
            list(set(os.listdir(root_dir)) - set(categories)),
        )
    )
    for category in other_categories:
        img_dir = os.path.join(root_dir, category)
        images = list(filter(lambda x: not x.startswith("."), os.listdir(img_dir)))
        sampled_images = random.sample(images, 30)
        for img in sampled_images:
            f.write(root_dir + "/" + os.path.join(category, img) + "\t" + str(5) + "\n")

with open("data/others.txt", "w") as f:
    other_categories = list(
        filter(
            lambda x: not x.startswith("."),
            list(set(os.listdir("data/test")) - set(categories)),
        )
    )
    root_dir = "data/test"
    for category in other_categories:
        img_dir = os.path.join(root_dir, category)
        images = list(filter(lambda x: not x.startswith("."), os.listdir(img_dir)))
        for img in images[:143]:
            f.write(root_dir + "/" + os.path.join(category, img) + "\t" + str(5) + "\n")
