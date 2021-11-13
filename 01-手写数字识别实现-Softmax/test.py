import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch

import mnist_data_loader

mnist_dataset = mnist_data_loader.read_data_sets("../DataSet/MNIST_data/", one_hot=True)

# training dataset
train_set = mnist_dataset.train
# 1-learning dataset
test_set = mnist_dataset.test

train_size = train_set.num_examples
test_size = test_set.num_examples
print()
print('Training dataset size: ', train_size)
print('Test dataset size: ', test_size)

batch_size = 100
max_epoch = 10
learning_rate = 0.001

# For regularization
lamda = 0.5

from softmax_classifier import softmax_classifier, load_pred_images

W = np.random.randn(28 * 28, 10) * 0.01
W = torch.from_numpy(W).to(torch.float32)
loss_set = []
accu_set = []
disp_freq = 100

for epoch in range(0, max_epoch):
    iter_per_batch = train_size // batch_size
    for batch_id in range(0, iter_per_batch):
        batch = train_set.next_batch(batch_size)  # get data of next batch
        input, label = batch
        # 全部换为tensor类型
        input = torch.from_numpy(input).to(torch.float32)
        label = torch.from_numpy(label).to(torch.float32)
        # softmax_classifier
        loss, gradient, prediction = softmax_classifier(W, input, label, lamda)
        # Calculate accuracy
        label = np.argmax(label, axis=1)  # scalar representation
        accuracy = sum(numpy.array(prediction) == numpy.array(label)) / float(len(label))

        loss_set.append(loss)
        accu_set.append(accuracy)

        # Update weights
        W = torch.from_numpy(np.asarray(W))
        W = W - (learning_rate * gradient)
        if batch_id % disp_freq == 0:
            print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
                epoch, max_epoch, batch_id, iter_per_batch,
                loss, accuracy))
    print()


test_imags, result = load_pred_images(W, False)
print(result)
for index, image in enumerate(test_imags):
    image = image.numpy()
    plt.imshow(np.reshape(image, [28, 28]))
plt.show()

#
# for index, image in enumerate(test_imag):
#     image = image.numpy()
#     print(result[index])
#     plt.imshow(np.reshape(image, [28, 28]))
