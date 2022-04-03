import os

categories = ['Bak Chor Mee', 'Char Kway Teow', 'Chicken Rice', 'Satay', 'Oyster Omelette']

with open('data/train.txt', 'w') as f:
    root_dir = 'data/train'
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        for img in os.listdir(img_dir):
            f.write(root_dir + '/' + os.path.join(folder, img) + '\t' + str(i) + '\n')

with open('data/val.txt', 'w') as f:
    root_dir = 'data/val'
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        for img in os.listdir(img_dir):
            f.write(root_dir + '/' + os.path.join(folder, img) + '\t' + str(i) + '\n')

with open('data/test.txt', 'w') as f:
    root_dir = 'data/test'
    for i, folder in enumerate(categories):
        img_dir = os.path.join(root_dir, folder)
        for img in os.listdir(img_dir):
            f.write(root_dir + '/' + os.path.join(folder, img) + '\t' + str(i) + '\n')

