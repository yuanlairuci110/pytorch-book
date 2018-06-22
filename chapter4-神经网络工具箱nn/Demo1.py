import torch
from torch import nn
from torch.autograd import Variable as V
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

to_tensor = ToTensor()  # img -> tensor
to_pil = ToPILImage()
lena = Image.open('imgs/lena.png')
lena.show()

input = to_tensor(lena).unsqueeze(0)

# 锐化卷积核
kernel = torch.ones(3, 3)/-9
kernel[1][1] = 1
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)

out = conv((input))
to_pil(out.data.squeeze(0)).show()

