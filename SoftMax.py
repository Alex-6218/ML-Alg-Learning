import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os

train_files_path = ''
test_file = ''
K = 10
n = 784
epochs = 0
inputX = np.zeros((1, n))   
labelsY = np.zeros((1, K))
if os.path.isfile("softmax_params.npz"):
    print("Loading existing parameters from softmax_params.npz")
    params = np.load("softmax_params.npz")
    weightsT = params["Theta"]
    biasB = params["b"]


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

choice = input("Test or Train? (t/T for train, anything else for test): ").strip().lower()
if choice == 't':
    weightsT = np.random.randn(n, K) * 0.01  # small random values
    biasB = np.zeros((1, K))                 # initialize biases to zero
    epochs = 0
    inputX, labelsY = load_images_from_folder('./MNIST_Data/trainingSet/trainingSet')


def prediction(x, weights):
    #create logits
    logitsZ = x @ weights + biasB # shape (m, K)
    predictions = np.array([np.clip(np.exp(logitsZ[logit, :] - np.max(logitsZ[logit, :])), 0, None) for logit in range(logitsZ.shape[0])])
    return predictions / np.sum(predictions, axis=1, keepdims=True)  # shape (m, K)

def grad_ThetaT(x, y, weights):
    gradient = np.zeros(weights.shape)  # shape (n, K)
    p = prediction(x, weights)
    gradient += x.T @ (y - p)
    return gradient / x.shape[0]

def grad_B(x, y):
    gradient = np.zeros(biasB.shape)  # shape (1, K)
    for i in range(x.shape[0]):
        p = prediction(x[i:i+1, :], weightsT)
        gradient += (y[i:i+1, :] - p)
    return gradient / x.shape[0]

def train():
    global inputX, labelsY, weightsT, biasB, epochs, choice
    #calculate number of batches to make
    num_batches = inputX.shape[0] // 64
    lr_b, lr_w = 0.01, 0.01 # learning rates for bias and weights

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
        prev_err = 0
        max_iters = 300
        iters = 0
        while abs(d_error) > 1e-7 and iters < max_iters:
            #calculate error for the minibatch with current weight matrix
            err = 0
            for m in range(len(minibatch)):
                for k in range(K):
                    if minibatch_labels[m, k] == 1:
                        err += minibatch_labels[m, k] * np.log(prediction(minibatch, weightsT)[m, k])
            err = -err / len(minibatch)

            #adjust weight based on gradient
            weightsT = lr_w * grad_ThetaT(minibatch, minibatch_labels, weightsT)
            #adjust bias based on gradient
            biasB = lr_b * grad_B(minibatch, minibatch_labels)
            #calculate new error and d_err
            if iters == 0:
                    d_error = err
            else:
                d_error = err - prev_err
                prev_err = err
            iters += 1

            print(f"Minibatch {i+1}/{num_batches}: shape {minibatch.shape}, error {err}")
            print(f"  Iteration {iters}/{max_iters}, d_error: {d_error}")

    #increment epoch, save trained parameters, and randomize data for retraining
    shuffled_indices = np.random.permutation(len(inputX))
    inputX = inputX[shuffled_indices]
    labelsY = labelsY[shuffled_indices]

    np.savez("softmax_params.npz", Theta=weightsT, b=biasB)
    epochs += 1
    continue_training = input(f"Epoch {epochs} complete! Press Enter to continue training, or type 'stop' to end: ").strip().lower()
    if continue_training != 'stop':
        train()
        choice = ''
if choice == 't':
    train()
else:
    print("Please provide a test file path: ")
    test_file = input().strip()
    if not os.path.isfile(test_file):
        print(f"File {test_file} does not exist.")
        
    else:
        img = Image.open(test_file).convert("L")  # grayscale
        img = img.resize((28,28))                 # force 28x28 if needed
        arr = np.array(img).astype(np.float32) / 255.0
        testX = arr.flatten().reshape(1, -1)      # flatten 28x28 → 784 and reshape to (1, 784)

        pred = prediction(testX, weightsT)
        predicted_label = np.argmax(pred, axis=1)[0]
        confidence = pred[0, predicted_label]

        print(f"Predicted label: {predicted_label} with confidence {confidence:.4f}")

        plt.imshow(arr, cmap='gray')
        plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.4f})")
        plt.axis('off')
        plt.show()