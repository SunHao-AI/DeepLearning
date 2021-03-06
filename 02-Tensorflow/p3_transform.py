import numpy
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 测试数据类型转换
if __name__ == "__main__":
    writer = SummaryWriter("../logs/02/p3")
    bee_path = "../DataSet/ANT-BEE_data/label方式1/train/bees/16838648_415acd9e3f.jpg"

    """
        PIL转tensor
    """
    print("PIL转tensor:")
    bee_PIL = Image.open(bee_path)
    # 方式一
    bee_tensor = transforms.ToTensor()(bee_PIL)
    print(f"\t方式一: bee_tensor 类型为 {type(bee_tensor)}")
    # 方式二
    trans_tensor = transforms.ToTensor()
    bee_tensor = trans_tensor(bee_PIL)
    print(f"\t方式二: bee_tensor 类型为 {type(bee_tensor)}")
    """
        tensor转PIL
    """
    print("tensor转PIL")
    bee_PIL1 = transforms.ToPILImage(bee_tensor)
    print(f"\tbee_PIL1 类型为 {type(bee_PIL1)}")
    """
        PIL转ndaray
    """
    print("PIL转ndaray")
    bee_ndarray = numpy.array(bee_PIL)
    print(f"\tbee_ndarray 类型为 {type(bee_ndarray)}")
    """
        ndarray转PIL
    """
    print("ndarray转PIL")
    bee_PIL = Image.fromarray(bee_ndarray)
    print(f"\tbee_PIL 类型为 {type(bee_PIL)}")

    """
        ndaray转tensor
    """
    print("ndaray转tensor")
    bee_tensor = torch.from_numpy(bee_ndarray)
    print(f" \t方式一: bee_tensor 类型为 {type(bee_tensor)}")
    bee_tensor = transforms.ToTensor()(bee_ndarray)
    print(f"\t方式二: bee_tensor 类型为 {type(bee_tensor)}")
    trans_tensor = transforms.ToTensor()
    bee_tensor = trans_tensor(bee_ndarray)
    print(f"\t方式三: bee_tensor 类型为 {type(bee_tensor)}")
    print("-" * 100)

    """
        ToTensor
    """
    image_path = "../DataSet/IMAGE_data/image1.jpg"
    image_PIL = Image.open(image_path)
    trans_tensor = transforms.ToTensor()
    image_tensor = trans_tensor(image_PIL)
    print(f"image_tensor 类型为 {type(image_tensor)}")

    """
         Normalize:  input[channel] = (input[channel] - mean[channel]) / std[channel]
         参数
         mean:均值,由于图片是RGB图片,有3通道,所有有3个均值
         std:标准差,.............................标准差
         inplace:是否原地操作

    """
    trans_norm = transforms.Normalize([0, 0.5, 0], [1, 1.5, 1])
    image_norm_tensor = trans_norm(image_tensor)
    writer.add_image("Normalize", image_norm_tensor)

    """
        Resize
    """
    print(f"原图: image_PIL.size = {image_PIL.size}")
    # 2个维度,指定宽高
    trans_resize = transforms.Resize((512, 512))
    image_resize_PIL1 = trans_resize(image_PIL)
    print(f"Resize(512, 512)后: image_PIL.size = {image_resize_PIL1.size}")
    # 1个维度,等比缩放
    trans_resize = transforms.Resize(300)
    image_resize_PIL2 = trans_resize(image_PIL)
    print(f"Resize(300)后: image_PIL.size = {image_resize_PIL2.size}")

    trans_tensor = transforms.ToTensor()
    writer.add_image("Resize", trans_tensor(image_resize_PIL1), 0)
    writer.add_image("Resize", trans_tensor(image_resize_PIL2), 1)

    """
        Compose
    """
    trans_resize = transforms.Resize((800, 50))
    trans_tensor = transforms.ToTensor()
    trans_compose = transforms.Compose(
        [
            trans_resize,
            trans_tensor
        ]
    )
    image_compose_tensor = trans_compose(image_PIL)
    writer.add_image("Resize", image_compose_tensor, 2)

    """
        RandomCrop
    """
    trans_crop = transforms.RandomCrop(400)
    trans_compose = transforms.Compose([trans_crop, trans_tensor])
    for i in range(10):
        image_crop_tensor = trans_compose(image_PIL)
        writer.add_image("RandomCrop", image_crop_tensor, i)
    writer.close()
