import os
from PIL import Image
import numpy as np
import torch


def softmax_func(x):
    x = torch.tensor(x)
    x -= x.max(1, keepdim=True).values
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition


def for_word(x, y, w, lamda):
    S = softmax_func(x @ w)
    x_rows = x.shape[0]
    c = -y * torch.log(S)
    loss = c.sum() / x_rows + 0.5 * lamda * torch.norm(w)
    dw = -1 / x_rows * (x.t() @ (y - S)) + lamda * w
    return loss, dw, S


def back_word(x, y, w, lamda):
    return for_word(x, y, w, lamda)


def predict(s):
    return np.argmax(s, axis=1)


def load_pred_images(w, black_background=True):
    if black_background:
        img_folder = "../DataSet/MNIST-自定义_data/black"
    else:
        img_folder = "../DataSet/MNIST-自定义_data/white"
    img_list = []
    imlist = os.listdir(img_folder)
    for imagename in imlist:
        im_url = os.path.join(img_folder, imagename)
        img = Image.open(im_url)
        img = np.asarray(img, dtype="float32")
        img_list.append(img.reshape(1, -1))
    x = np.array(img_list).reshape(-1, 784)
    x = torch.tensor(x)
    label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    label = torch.from_numpy(label)
    _, _, prediction = softmax_classifier(w, x, label, 0.5)
    s1 = x @ w
    pred = torch.max(s1, 1)
    data = torch.Tensor(img_list).to(torch.float32).reshape(-1, 784)
    return data, pred


def test_verify(data, w):
    s = softmax_func(data @ w)
    return predict(s)


def softmax_classifier(W, input, label, lamda):
    ############################################################################
    loss, gradient, s = back_word(input, label, W, lamda)
    prediction = predict(s)
    ############################################################################
    return loss, gradient, prediction

# def softmax(x):
#     x_exp = np.exp(x)
#     partition = x_exp.sum(1, keepdim=True)
#     return x_exp / partition
#
#
# def softmax_classifier(W, input, label, lamda):
#     """
#     :param W: ndarray,784*100
#     :param input: ndarray,100*784
#     :param label: ndarray,100*10
#     :param lamda: float
#     :return:
#     """
#     ############################################################################
#     # TODO: Put your code here
#     gradient = None
#     # 1.转换为torch.tensor
#     W, input, label = torch.tensor(W).to(torch.float32), \
#                       torch.tensor(input).to(torch.float32), \
#                       torch.tensor(label).to(torch.int)
#
#     # 2.f(w,x) = X[100*784] @ W[784*10]
#     f = input @ W
#     # 3.softmax
#     #   3.1 f每一项减去矩阵最大值，防止过溢
#     # f -= torch.max(f, 1).values.reshape(-1, 1)
#     f = f - torch.max(f)
#     #   3.2 softmax
#     s = torch.exp(f) / torch.exp(f).sum(1, keepdim=True)
#     # 4 交叉熵
#     c = - torch.log(s) * label
#     # 5.损失函数loss
#     loss = (c.sum() + 0.5 * lamda * torch.norm(W) ** 2) / len(input)
#     # 6.预测
#     prediction = np.argmax(s, axis=1)
#     # pred = torch.zeros_like(s)
#     # index = torch.max(s, 1)
#     # for i in range(pred.shape[0]):
#     #     pred[i, index.indices[i]] = 1
#     # prediction = np.argmax(pred, axis=1)  # scalar representation
#     # 7.求导
#     gradient = - input.t() @ ((1 - s) * label) + lamda * W
#     ############################################################################
#     return loss, gradient, prediction

# def softmax_classifier(W, input, label, lamda):
#     """
#     :param W: ndarray,784*100
#     :param input: ndarray,100*784
#     :param label: ndarray,100*10
#     :param lamda: float
#     :return:
#     """
#     ############################################################################
#     # TODO: Put your code here
#     gradient = None
#     # 1.转换为torch.tensor
#     W, input, label = torch.tensor(W).to(torch.float32), \
#                       torch.tensor(input).to(torch.float32), \
#                       torch.tensor(label).to(torch.int)
#
#     # 2.f(w,x) = X[100*784] @ W[784*10]
#     f = input @ W
#     # 3.softmax
#     #  3.1 f每一项减去当前行最大值，防止过溢
#     # f -= torch.max(f, 1).values.reshape(-1, 1)
#     f -= torch.max(f)
#     #  3.2 softmax
#     s = torch.exp(f) / torch.exp(f).sum(1, keepdim=True)
#     # 4 交叉熵
#     c = - torch.log(s) * label
#     # 5.损失函数loss
#     loss = c.sum() + lamda * torch.norm(W) ** 2 / 2
#     # 6.预测
#     pred = torch.zeros_like(s)
#     index = torch.max(s, 1)
#     for i in range(pred.shape[0]):
#         pred[i, index.indices[i]] = 1
#     prediction = np.argmax(pred, axis=1)  # scalar representation
#     # 7.求导
#     gradient = -input.t() @ ((label / s) * (s - s ** 2)) +lamda * W
#     ############################################################################
#     return loss, gradient, prediction
