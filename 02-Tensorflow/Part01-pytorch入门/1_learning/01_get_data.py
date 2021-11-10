import os.path
import typing

import PIL
from torch.utils.data import Dataset
from PIL import Image

"""
本单元知识点:
    1.os.path.join(dir1, dir2,....)——将dir1、dir2、...拼接成路径字符串,适用于各操作系统,不用担心windows和linux路径不同问题
    2.os.listdir(文件夹)——将文件夹下的文件名存入列表
    3.Image.open(路径/xx.jpg)——读取图片数据,返回一个PIL.Image对象
    4.PIL.Image.show(路径/xx.jpg)——展示图片
    
"""


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        """ 数据集路径: dataset
                        ->train
                            ->ants
                            ->bees
        :param root_dir:    dataset\train
        :param label_dir:   ants or bees
        : path_dir:    dataset\train\ants
        : img_path:    利用os.listdir()将路径dataset\train\ants下的图片名称加入列表
                       如:img_list = [0013035.jpg,5650366_e22b7e1065.jpg,...]
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index) -> typing.Tuple[PIL.Image.Image, str]:
        img_name = self.img_path[index]
        img_iter_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img_data = Image.open(img_iter_path)
        img_label = self.label_dir
        return img_data, img_label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    test_data = MyData("../dataset/train", "ants")
    print(len(test_data))
    test_data, label = test_data[3]
    test_data.show()
