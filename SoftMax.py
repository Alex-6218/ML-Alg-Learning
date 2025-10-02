import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os

train_files_path = ''
test_file = ''
K = 10
n = 784

def load_images_from_folder(folder, num_classes=10, img_size=(28,28)):
    data, labels = [], []
    for label in range(num_classes):
        path = os.path.join(folder, str(label))
        print(f"Loading images from: {path}")
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                print(f"Processing file: {filename}")
                img_path = os.path.join(path, filename)
                img = Image.open(img_path).convert("L")  # grayscale
                img = img.resize(img_size)               # force 28x28 if needed
                arr = np.array(img).astype(np.float32) / 255.0
                data.append(arr.flatten())               # flatten 28x28 → 784
                labels.append(label)
    return np.array(data), np.array(labels)

x_train, y_train = load_images_from_folder('./MNIST_Data/trainingSet/trainingSet')

for var, val in [('x_train', x_train), ('y_train', y_train)]:
    print(f"{var}: shape {val.shape}, dtype {val.dtype}")

def cost(predictions, targets):
    loss = 0
    for i in range(len(targets)):
        for k in range(K):
            if predictions[i] == k:
                loss += np.log(predictions[i][k])
    return -loss
 