import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os

train_files_path = ''
test_file = ''
K = 10
n = 784


inputX = np.zeros((1, n))   
labelsY = np.zeros((1, K))
weightsT = np.zeros((n, K))

def load_images_from_folder(folder, img_size=(28,28)):
    data, labels = [], []
    for label in range(K):
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
                #create one-hot encoded label
                label_one_hot = np.zeros(K)
                label_one_hot[label] = 1
                labels.append(label_one_hot)
    return np.array(data), np.array(labels)

inputX, labelsY = load_images_from_folder('./MNIST_Data/trainingSet/trainingSet')

for var, val in [('inputX', inputX), ('labelsY', labelsY)]:
    print(f"{var}: shape {val.shape}, dtype {val.dtype}")


def prediction(x, weights):
    predictions = []
    return predictions

def train():
    global inputX, labelsY, weightsT
    #calculate number of batches to make
    num_batches = inputX.shape[0] // 64
    epochs = 0
    print(f"Number of batches: {num_batches}")

    #create minibatches of size 64 and train on each minibatch
    for i in range(num_batches):
        minibatch = inputX[64*i:64*(i+1), :]
        #create corresponding labels for minibatch
        minibatch_labels = np.zeros((64, K))
        for l in range(len(minibatch)):
            minibatchlabel = labelsY[64*i + l, :]
            minibatch_labels[l, :] = minibatchlabel

        d_error = 10
        max_iters = 300
        iters = 0
        while abs(d_error) > 1e-3 and iters < max_iters:
            #calculate error for the minibatch with current weight matrix
            err = 0
            for m in range(len(minibatch)):
                for k in range(K):
                    if minibatch_labels[m, k] == 1:
                        err += np.log(prediction(minibatch[m, :], weightsT)[k])
            #adjust weights based on gradient 
            #calculate new error and d_err
        print(f"Minibatch {i+1}/{num_batches}: shape {minibatch.shape}, error {err}")

    #increment epoch and randomize data for retraining
    shuffled_indices = np.random.permutation(len(inputX))
    inputX = inputX[shuffled_indices]
    labelsY = labelsY[shuffled_indices]
    epochs += 1

train()


