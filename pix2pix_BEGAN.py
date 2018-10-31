import argparse

import math
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.transforms import transforms

from dataset import get_coco_loader, get_folder_loader, toGray
from dexter import log_variable
from image_pool import ImagePool
from logger import Logger
from network import UnetGenerator, NLayerDiscriminator, GANLoss, UnetDescriminator, init_net

import socket
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

lr = 0.0002
beta1 = 0.5
lambda_L1 = 100.0
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
no_lsgan = False
epocs = 5
pool_size = 50
image_size = 256

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='128_crop',  help='path to dataset')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nz', type=int, help='size of random vector', default=64)
parser.add_argument('--imsize', type=int, help='size of image', default=128)
parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--comment', help='comments', default='')

opt = parser.parse_args()


class BEGANModel():
    def __init__(self):
        self.kt = 0
        self.lamk = 0.001

        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        self.netD = UnetDescriminator(input_nc=6, output_nc=6, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        init_net(self.netG, 'normal', 0.002, [0])
        init_net(self.netD, 'normal', 0.002, [0])

        import os
        if os.path.isfile('checkpoints/netD' + socket.gethostname() + '.pth'):
            print('Loading model....')
            self.netG = torch.load('checkpoints/netG' + socket.gethostname() + '.pth')
            self.netD = torch.load('checkpoints/netD' + socket.gethostname() + '.pth')

        self.netG.to(self.device)
        self.netD.to(self.device)


    # criterionGAN = GANLoss(use_lsgan=not no_lsgan).to(device)
    # criterionL1 = torch.nn.L1Loss()

    # initialize optimizers
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

        if opt.workers==0:
            expname = ''
        else:
            expname = '-'.join(['b_'+str(opt.batchSize), 'nz_'+str(opt.nz), 'gm_'+str(opt.gamma)])

        self.writer = Logger('runs/'+socket.gethostname()+'-'+datetime.now().strftime('%B%d-%H-%M-%S')+expname+opt.comment)

    def set_input(self, data):
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)

    def L_Df(self, v):
        reconstructed = self.netD(v)
        return (reconstructed-v).abs().mean()

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_G(self):
        self.optimG.zero_grad()
        self.xG = torch.cat([self.real_A, self.fake_B], 1)
        self.d_fake = self.L_Df(self.xG)
        self.d_fake.backward(retain_graph=True)
        self.optimG.step()

    def backward_D(self):
        self.optimD.zero_grad()
        self.xR = torch.cat([self.real_A, self.real_B], 1)
        self.d_real = self.L_Df(self.xR)
        self.L_D = self.d_real - self.kt * self.d_fake
        self.L_D.backward()
        self.optimD.step()
        self.L_D_val = self.L_D.item()
        self.L_G_val = self.d_fake.item()

    def update_K(self):
        self.kt = self.kt + self.lamk * (opt.gamma * self.L_D_val - self.L_G_val)
        if self.kt < 0:
            self.kt = 0
        self.M_global = self.L_D_val + math.fabs(opt.gamma * self.L_D_val - self.L_G_val)

    def log(self, n_iter):
        LD_LG = self.L_D_val - self.L_G_val
        print('M_global: {}, L_D_val: {}, L_G_val: {}, kt: {}, LD_LG: {}'
              .format(self.M_global, self.L_D_val, self.L_G_val, self.kt, LD_LG))

        if n_iter % 10000 == 0:
            opt.lr = opt.lr / 2
            for param_group in self.optimD.param_groups:
                param_group['lr'] = opt.lr  # param_group['lr']/2
            for param_group in self.optimG.param_groups:
                param_group['lr'] = opt.lr  # param_group['lr']/2

        if n_iter % 1000 == 1:
            self.writer.scalar_summary('misc/M_global', self.M_global, n_iter)
            self.writer.scalar_summary('misc/kt', self.kt, n_iter)
            self.writer.scalar_summary('loss/L_D', self.L_D_val, n_iter)
            self.writer.scalar_summary('loss/L_G', self.L_G_val, n_iter)
            self.writer.scalar_summary('loss/d_real', self.d_real.item(), n_iter)
            self.writer.scalar_summary('loss/d_fake', self.d_fake.item(), n_iter)


            test_A = self.test_data['A']
            test_B = self.test_data['B']

            with torch.no_grad():
                fake_B = self.netG(test_A.to(self.device))
            self.writer.scalar_summary('misc/learning', opt.lr, n_iter)
            images = torch.cat([test_A, test_B, fake_B.cpu()])
            x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=4)
            self.writer.image_summary('Input / Real / Generated', [x], n_iter)

            torch.save(self.netD, 'checkpoints/netD' + socket.gethostname() + '.pth')
            torch.save(self.netG, 'checkpoints/netG' + socket.gethostname() + '.pth')
            for name, param in self.netG.named_parameters():
                if 'bn' in name:
                    continue
                self.writer.histo_summary('weight_G/' + name, param.clone().cpu().data.numpy(), n_iter)
                self.writer.histo_summary('grad_G/' + name, param.grad.clone().cpu().data.numpy(), n_iter)

            for name, param in self.netD.named_parameters():
                if 'bn' in name:
                    continue
                self.writer.histo_summary('weight_D/' + name, param.clone().cpu().data.numpy(), n_iter)
                self.writer.histo_summary('grad_D/' + name, param.grad.clone().cpu().data.numpy(), n_iter)

    def set_test_input(self, test_data):
        self.test_data = test_data


if __name__ == '__main__':
    print(opt)
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Lambda(toGray),
    ])

    dataloader = get_folder_loader(dataroot=opt.dataroot, transform=tf, batch_size=4, shuffle=True, num_workers=1)

    model = BEGANModel()

    test_data, _ = next(iter(dataloader))
    model.set_test_input(test_data)
    n_iter = 0
    for epoch in range(epocs):
        for i, (data, labels) in enumerate(dataloader):
            # n_iter = i + epoch * len(dataloader)
            model.set_input(data)

            model.forward()
            model.backward_G()
            model.backward_D()
            model.update_K()

            model.log(n_iter)
            n_iter += 1







