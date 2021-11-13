import torchvision.datasets

# 下载数据集
train_set = torchvision.datasets.CIFAR10(root="../DataSet/CIFAR10", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="../DataSet/CIFAR10", train=False, download=True)
