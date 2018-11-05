import argparse

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.transforms import transforms
from dataset import get_folder_loader
from image_pool import ImagePool
from imageutils import distort, todict
from logger import Logger
from network import UnetGenerator, NLayerDiscriminator, GANLoss, init_net
import os
import shutil
from datetime import datetime

beta1 = 0.5
lsgan = False
epocs = 2000
pool_size = 50

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='128_crop',  help='path to dataset')
parser.add_argument('--valroot',  help='path to validation dataset')
parser.add_argument('--rundir', default='runs',  help='path to logs and models')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--ngf', type=int, help='generator filter size', default=128)
parser.add_argument('--ndf', type=int, help='descriminator filter size', default=128)
parser.add_argument('--imsize', type=int, help='size of image', default=256)
parser.add_argument('--lambdaL1', type=float, help='gamma', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--saveinterval', type=int, help='learning rate', default=500)
parser.add_argument('--loginterval', type=int, help='learning rate', default=2000)
parser.add_argument('--lrdecayInterval', type=int, help='learning rate', default=10000)
parser.add_argument('--comment', help='comments', default='')

opt = parser.parse_args()


class Pix2PixModel():
    def __init__(self, gpu_ids=[0], continue_run=None, epoch='latest'):
        self.model_names = ['netD', 'netG']
        self.gpu_ids = gpu_ids

        # Decide which device we want to run on
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0]) if (torch.cuda.is_available()) else "cpu")

        self.netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=5, ngf=opt.ngf, norm_layer=nn.BatchNorm2d, use_dropout=True).to(self.device)
        init_net(self.netG, 'normal', 0.002, [0])

        self.netD = NLayerDiscriminator(input_nc=6, ndf=opt.ndf, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=True).to(self.device)
        init_net(self.netD, 'normal', 0.002, [0])

        self.fake_AB_pool = ImagePool(pool_size)

        self.criterionGAN = GANLoss(use_lsgan=lsgan).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(beta1, 0.999))

        if not continue_run:
            expname = '-'.join(['b_' + str(opt.batchSize), 'ngf_' + str(opt.ngf), 'ndf_' + str(opt.ndf), 'gm_' + str(opt.lambdaL1)])
            self.rundir = opt.rundir + '/pix2pixGAN-' + datetime.now().strftime('%B%d-%H-%M-%S') + expname + opt.comment
        else:
            self.rundir = continue_run
            self.load_networks(epoch)

        self.writer = Logger(self.rundir)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * opt.lambdaL1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def backward(self):
        set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def set_input(self, data):
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)

    def set_test_input(self, test_data):
        self.test_data = test_data

    def set_val_input(self, val_data):
        self.val_data = val_data


    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def visualize(self, epoc, i_batch, step):
        self.writer.scalar_summary('loss_D_fake', self.loss_D_fake, step)
        self.writer.scalar_summary('loss_D_real', self.loss_D_real, step)
        self.writer.scalar_summary('loss_D', self.loss_D, step)
        self.writer.scalar_summary('loss_G_GAN', self.loss_G_GAN, step)
        self.writer.scalar_summary('loss_G_L1', self.loss_G_L1, step)
        self.writer.scalar_summary('loss_G', self.loss_G, step)

        test_A = self.test_data['A']
        test_B = self.test_data['B']

        val_A = self.val_data['A']

        with torch.no_grad():
            fake_test_B = self.netG(test_A.to(self.device))
            fake_val_B = self.netG(val_A.to(self.device))

        self.writer.scalar_summary('misc/learning', opt.lr, step)

        images = torch.cat([test_A, test_B, fake_test_B.cpu()])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=opt.batchSize)
        self.writer.image_summary('Test_Fixed', [x], step)

        images = torch.cat([self.real_A.detach().cpu(), self.real_B.detach().cpu(), self.fake_B.detach().cpu()])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=opt.batchSize)
        self.writer.image_summary('Test_Last', [x], step)

        images = torch.cat([val_A, fake_val_B.cpu()])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=opt.batchSize)
        self.writer.image_summary('Validation', [x], step)

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{}_{}.pth'.format(epoch, name)
                save_path = os.path.join(self.rundir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                self.update_link(save_path, os.path.join(self.rundir, 'latest_{}.pth'.format(name)))

    def update_link(self, src, dst):
        # try:
        #     os.symlink(src, dst)
        # except OSError as err:
        shutil.copy2(src, dst)

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (epoch, name)
                load_path = os.path.join(self.rundir, load_filename)
                if not os.path.isfile(load_path):
                    return

                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def updatelr(self):
        opt.lr = opt.lr / 2
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = opt.lr  # param_group['lr']/2
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = opt.lr  # param_group['lr']/2


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.Resize(opt.imsize),
        transforms.CenterCrop(opt.imsize),
        transforms.Lambda(distort),
    ])
    dataloader = get_folder_loader(dataroot=opt.dataroot, transform=tf, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    if opt.valroot:
        val_tf = transforms.Compose([
            transforms.Resize(opt.imsize),
            transforms.CenterCrop(opt.imsize),
            transforms.Lambda(todict),
        ])
        valloader = get_folder_loader(dataroot=opt.valroot, transform=val_tf, batch_size=opt.batchSize, shuffle=True, num_workers=1)
    else:
        valloader = dataloader

    # model = Pix2PixModel(continue_run='runs/pix2pixGAN-November09-16-05-06b_8-ngf_192-ndf_128-gm_0.5', epoch='27')
    model = Pix2PixModel()
    test_data, _ = next(iter(dataloader))
    model.set_test_input(test_data)

    model.load_networks('latest')

    decay_period = opt.lrdecayInterval
    step = 43000
    for epoc in range(epocs):
        for i_batch, (data, labels) in enumerate(dataloader):
            model.set_input(data)
            model.forward()
            model.backward()

            print('lossD_fake: {}, lossD_real: {}, lossG_GAN: {}, lossG_L1: {}, loss_G: {}'.format(model.loss_D_fake, model.loss_D_real, model.loss_G_GAN, model.loss_G_L1, model.loss_G))

            if step % opt.loginterval == 0:
                test_data, _ = next(iter(valloader))
                model.set_val_input(test_data)
                model.visualize(epoc, i_batch, step)

            if step % opt.saveinterval == 0:
                model.save_networks(epoc)

            if step % decay_period == 0:
                model.updatelr()
                decay_period += opt.lrdecayInterval
            step += opt.batchSize







