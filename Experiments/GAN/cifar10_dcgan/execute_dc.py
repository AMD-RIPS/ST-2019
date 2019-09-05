import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab
import numpy as np


num_gpu = 1 if torch.cuda.is_available() else 0

# load the models
from dcgan import Discriminator, Generator

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

D = D.double()

# load weights
D.load_state_dict(torch.load('weights/netD_epoch_199.pth', map_location='cpu'))
G.load_state_dict(torch.load('weights/netG_epoch_199.pth', map_location='cpu'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()


batch_size = 25
latent_size = 100

fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
if torch.cuda.is_available():
    fixed_noise = fixed_noise.cuda()
fake_images = G(fixed_noise)


# z = torch.randn(batch_size, latent_size).cuda()
# z = Variable(z)
# fake_images = G(z)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 3, 32, 32)
fake_images_np = fake_images_np.transpose((0, 2, 3, 1))
R, C = 5, 5
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.imshow(fake_images_np[i], interpolation='bilinear')
plt.show()


# outputs = D(fake_images)
# print(outputs)



new_imgs = np.random.uniform(0,1,size= (25,32,32,3))

p = torch.from_numpy(new_imgs.transpose(0,3,1,2))


outputs = D(p)
print(outputs)
