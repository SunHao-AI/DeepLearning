# 图片读取



## 1.	方式一： PIL.Image.Image.open("图片的路径")读取图片

```python
from PIL import Image
# Image.open("图片的路径")返回一个PIL.JepgImage类型的数据对象
图片数据_PIL = Image.open("图片的路径")
# 将PIL.JepgImage对象转换为ndarray
图片数据_ndarray = numpy.array(图片数据_PIL)
```



## 2.	方式二：cv2.imread("图片的路径")读取图片)

```python
import cv2
# cv.imread("图片的路径")返回一个ndarray类型的数据对象
图片数据_ndarray = cv.imread("图片的路径")
```





<span id="#数据转换"></span>

# 数据转换 

## 1.	PIL<==>ndarray

### 	1.1	PIL==>ndarray

```python
data_ndarray = numpy.array(data_PIL)
```

### 	1.2.	ndarray==>PIL

```python
data_PIL = Image.fromarray(data_ndarray)
```



## 2.	PIL<==>tensor

### 	2.1	PIL，ndarray==>tensor

```python
# 方式一
data_tensor = transforms.ToTensor()(data_ndarrayOrPIL)
# 方式二
trans_tensor = transforms.ToTensor()
bee_tensor2 = trans_tensor(bee_PIL)
```

### 	2.2	tensor==>PIL

```python
data_PIL = transforms.ToPILImage(data_tensor)
```



## 3.	ndarray<==>tensor

### 	3.1	ndarray==>tensor

```python
# 方式一
data_tensor = torch.from_numpy(data_ndarray)
# 方式二.1
data_tensor = transforms.ToTensor()(data_ndarray)
# 方式二.2
```

### 	3.2	tensor==>ndarray

```python
data_ndarray = data_tensor.numpy()
```



# tensorboard

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs路径")
```

## 1. SummaryWriter简单用法

### 	1.1	writer.add_scalar()

```python
"""
writer.add_image()
Args: 
    tag (str) : 数据标识符——标题
    scalar_value (Any) : 要保存的值——f(x)函数值，因变量y
    global_step (int=None) : 要记录的全局步长值——f(x)函数自变量x
    walltime (float=None) : 可选覆盖默认的 walltime (time.time()) 与事件纪元后的秒数
    new_style (bool=False) : 是使用新样式（张量字段）还是旧样式（simple_value 字段）。 新样式可能会导致更快的数据加载。
"""
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs路径")
for i in range(1, 11):
    writer.add_scalar('quadratic', i ** 2, global_step=i)
    writer.add_scalar('exponential', 2 ** i, global_step=i)
writer.close()
```

### 	1.2	writer.add_image（）

```python
"""
writer.add_image（）
Args: 
    tag (str) : 数据标识符——标题
    img_tensor [Any(tensor,ndarry,stringblobname)] : 图像数据
    global_step (int=None) : 要记录的全局步长值——f(x)函数自变量x
    walltime (float=None) : 事件发生后的可选覆盖默认 walltime (time.time()) 秒
    dataformats (str='CHW')
        CHW：通道数、高、宽，常用HWC，要和img_tensor.shape对应
"""
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs路径")
# 读取图片数据
image_path = "图片路径"
image_PIL = Image.open(image_path)
# 转为tensor数据
image_tensor = transforms.ToTensor()(image_PIL)
# 写入tensorboard
writer.add_image("ants", ant_array, i, dataformats="HWC")
```





# transforms

## 1.	数据转换

​		[点击跳转](#数据转换)

## 2.	ToTensor

```python
"""
转换范围内的 PIL Image 或 numpy.ndarray (HWC) [0, 255] 到一个 Torch.FloatTensor，其形状 (CHW) 在 [0.0, 1.0] 范围内.
如果 PIL 图像属于其中一种模式（L、LA、P、I、F、RGB、YCbCr、RGBA、CMYK、1） 或者是 numpy.ndarray数据，有 dtype = np.uint8  
在其他情况下，张量在没有缩放的情况下返回。

__call__：
    Args:
        pic (PIL Image or numpy.ndarray): PIL、ndarray==>tensor

    Returns:
        tensor: tensor数据
"""
from PIL import Image
from torchvision import transforms

image_path = "图片路径"
image_PIL = Image.open(image_path)
# 方式一
trans_tensor = transforms.ToTensor()
image_tensor = trans_tensor(image_PIL)
# 方式二
trans_tensor = transforms.ToTensor()(image_PIL)

print(type(image_tensor))
```

## 3.	Normalize

```python
"""
使用均值和标准差对张量图像进行标准化。
给定均值：(M1,...,Mn) 和 std: (S1,..,Sn) 用于 ``n`` 通道，此变换将标准化输入的每个通道 torch.Tensor
即 input[channel] = (input[channel] - mean[channel]) / std[channel] 
注意::此变换行为不合适，即它不会改变输入张量。
Args：
	mean （序列）：每个通道的均值序列。
	std （序列）：每个通道的标准偏差序列。

__call__：
    Args:
       tensor (Tensor): 尺寸为(C, H, W)的tensor数据

	Returns:
		tensor: 正则化的tensor数据
	
"""
from PIL import Image
from torchvision import transforms

image_path = "图片路径"
image_PIL = Image.open(image_path)

trans_norm = transforms.Normalize([0, 0.5, 0], [1, 1.5, 1])
# 必须是tensor数据类型
image_norm_tensor = trans_norm(image_tensor)
```

## 4.	Resize

```python
"""
将输入的 PIL Image 调整为给定的大小。
Args: 
	size (sequence or int): 期望的输出大小。如果 size 是像 (h, w) 这样的序列，则输出大小将与此匹配。
如果 size 是一个int，图像的较小边缘将与此数字匹配。即，如果高度 > 宽度，则图像将重新缩放为（size * height / width, size）

__call__:
    Args:
    	img (PIL Image): image_PIL

    Returns:
    	PIL Image: image_PIL
"""
from PIL import Image
from torchvision import transforms

image_path = "图片路径"
image_PIL = Image.open(image_path)

# 1个维度，选短边进行等比缩放
trans_resize = transforms.Resize(300)
# 必须是PIL类型
image_norm_PIL = trans_resize(image_PIL)
print(image_norm_PIL.size)

# 2个维度,指定宽高缩放
trans_resize = transforms.Resize((512, 600))
# 必须是PIL类型
image_norm_PIL = trans_resize(image_PIL)
print(image_norm_PIL.size)
```

## 5.	RandomCrop

```python
"""
在随机位置裁剪给定的 PIL 图像。
Args: 
	size (sequence or int): 期望的裁剪输出尺寸。如果 size 是 int 而不是像 (h, w) 这样的序列，则为方形裁剪（size,尺寸）填充。
	padding（int or sequence，可选）：图像的每个边框上的可选填充。默认为无，即无填充。如果提供长度为4的序列，则用于填充左、上、右, 分别下边框. 如果提供长度为 2 的序列, 则分别用于填充 leftright, topbottom 边框 
	pad_if_needed (boolean): 如果小于所需大小，则填充图像以避免引发异常
	fill：常量填充的像素填充值。默认为0。如果是长度为3的元组，则分别用于填充R、G、B通道。该值仅在padding_mode为常量时使用
	padding_mode：填充类型。应为：常量、边缘、反射或对称。默认为常量。 
		- constant: 填充具有常量值，该值由填充指定
		- edge: 在图像边缘用最后一个值填充
		- reflect: 带有图像反射的垫块（不重复边缘上的最后一个值）
			在反射模式下用 2 个元素填充 [1, 2, 3, 4] 将导致 [3, 2, 1, 2, 3, 4, 3, 2]
		- symmetric: 带有图像反射的垫块(在边缘上重复最后的值)
            在对称模式下用 2 个元素填充 [1, 2, 3, 4] 将导致 [2, 1, 1, 2, 3, 4, 4, 3]
            
__call__:
	Args:
		img (PIL Image): image_PIL

	Returns:
		PIL Image: image_PIL
"""
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs路径")
from PIL import Image
from torchvision import transforms

image_path = "图片路径"
image_PIL = Image.open(image_path)

trans_tensor = transforms.ToTensor()
trans_crop = transforms.RandomCrop(400)
image_compose = transforms.Compose([trans_crop, trans_tensor])
for i in range(10):
    image_crop_PIL = image_compose(image_PIL)
    writer.add_image("RandomCrop", image_crop_PIL, i)
```

## 6.	Compose

```python
"""
将多个变换组合在一起。
Args:
	transforms (list of ``Transform`` objects): 组成变换的列表
          
Example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>> ])
__call__：
    Args：
        image_PIL
	Returns:
		image_PIL
"""
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs路径")
from PIL import Image
from torchvision import transforms

image_path = "图片路径"
image_PIL = Image.open(image_path)

trans_tensor = transforms.ToTensor()
trans_norm = transforms.Normalize([0, 0.5, 0], [1, 1.5, 1])
trans_resize = transforms.Resize((800, 50))
trans_crop = transforms.RandomCrop(400)
trans_compose = transforms.Compose(
    [
        trans_resize,	#PIL
        trans_crop,		#PIL
        trans_tensor,	#tensor
        trans_norm		#tensor
    ]
)
image_compose_tensor = trans_compose(image_PIL)
print(image_compose)
```



























