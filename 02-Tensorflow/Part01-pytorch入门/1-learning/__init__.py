import numpy
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

bee_path = "../../DataSet/dataset/train/bees/16838648_415acd9e3f.jpg"
image_PIL = Image.open(bee_path)
trans_tensor = transforms.ToTensor()
trans_norm = transforms.Normalize([0, 0.5, 0], [1, 1.5, 1])
trans_resize = transforms.Resize((512, 512))
trans_crop = transforms.RandomCrop(400)
trans_compose = transforms.Compose(
    [
        trans_resize,  # PIL
        trans_crop,  # PIL
        trans_tensor,  # tensor
        trans_norm  # tensor
    ]
)

image_compose = trans_compose(image_PIL)
print(image_compose)
