import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision.transforms import transforms

from dataset import get_coco_loader, get_folder_loader, toGray
from image_pool import ImagePool
from network import UnetGenerator, NLayerDiscriminator, GANLoss

image_dir = 'z:/Dataset/coco/val/orig/'
# ann_file = '/home/alex/Datasets/coco/annotations/instances_val2017.json'

lr = 0.0002
beta1 = 0.5
lambda_L1 = 100.0
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
no_lsgan = False
epocs = 5
pool_size = 50
image_size = 256


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
netD = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False)

fake_AB_pool = ImagePool(pool_size)

criterionGAN = GANLoss(use_lsgan=not no_lsgan).to(device)
criterionL1 = torch.nn.L1Loss()

# initialize optimizers
optimizers = []
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizers.append(optimizer_G)
optimizers.append(optimizer_D)

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.Lambda(toGray),
])
dataloader = get_folder_loader(dataroot=image_dir, transform=tf, batch_size=4, shuffle=True, num_workers=1)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def backward_D():
    # Fake
    # stop backprop to the generator by detaching fake_B
    fake_AB = fake_AB_pool.query(torch.cat((real_A, fake_B), 1))
    pred_fake = netD(fake_AB.detach())
    loss_D_fake = criterionGAN(pred_fake, False)

    # Real
    real_AB = torch.cat((real_A, real_B), 1)
    pred_real = netD(real_AB)
    loss_D_real = criterionGAN(pred_real, True)

    # Combined loss
    loss_D = (loss_D_fake + loss_D_real) * 0.5

    loss_D.backward()


def backward_G():
    # First, G(A) should fake the discriminator
    fake_AB = torch.cat((real_A, fake_B), 1)
    pred_fake = netD(fake_AB)
    loss_G_GAN = criterionGAN(pred_fake, True)

    # Second, G(A) = B
    loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1

    loss_G = loss_G_GAN + loss_G_L1

    loss_G.backward()


if __name__ == '__main__':

    for epoc in range(epocs):

        for i_batch, (data, labels) in enumerate(dataloader):
            # model.set_input(data)
            # model.optimize_parameters()

            real_A = data['A'].to(device)
            real_B = data['B'].to(device)

            fake_B = netG(real_A)

            set_requires_grad(netD, True)
            optimizer_D.zero_grad()
            backward_D()
            optimizer_D.step()

            # update G
            set_requires_grad(netD, False)
            optimizer_G.zero_grad()
            backward_G()
            optimizer_G.step()









