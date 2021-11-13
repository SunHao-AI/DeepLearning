import PIL
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

if __name__ == '__main__':
    writer = SummaryWriter("../logs/02/p2")
    # 读取一张图片
    image_path = "../DataSet/ANT-BEE_data/label方式1/train/ants/0013035.jpg"
    image_PIL = Image.open(image_path)
    # # image_PIL类型为PIL.JpegImagePlugin.JpegImageFile
    # print(type(image_PIL))
    image_array = np.array(image_PIL)
    # # (512, 768, 3) -> (HWC):高、宽、通道数
    # print(image_array.shape)
    # 将image_array对象的shape通过dataformats="HWC"属性告诉tensorboard
    writer.add_image("ant", image_array, 0, dataformats="HWC")

    # 读取文件夹内所有图片
    from p1_自定义数据集 import MyData

    ants_path = MyData("../DataSet/ANT-BEE_data/label方式1/train", "ants")
    for i in range(len(ants_path)):
        ant_PIL, _ = ants_path[i]
        # 只处理jpeg类型的图片
        if type(ant_PIL) == PIL.JpegImagePlugin.JpegImageFile:
            ant_array = np.array(ant_PIL)
            writer.add_image("ants", ant_array, i, dataformats="HWC")

    for i in range(1, 11):
        writer.add_scalar('quadratic', i ** 2, global_step=i)
        writer.add_scalar('exponential', 2 ** i, global_step=i)
    writer.close()
