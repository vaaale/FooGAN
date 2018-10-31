import torch
import torch.nn as nn
from torchvision.transforms import transforms
from dataset import get_coco_loader, get_folder_loader, toGray
from image_pool import ImagePool
from imageutils import distort
from logger import Logger
from network import UnetGenerator, NLayerDiscriminator, GANLoss, init_net
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

image_dir = 'y:/Dataset/B'
# ann_file = '/home/alex/Datasets/coco/annotations/instances_val2017.json'

lr = 0.0002
beta1 = 0.5
lambda_L1 = 100.0
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
no_lsgan = False
epocs = 200
pool_size = 50
image_size = 256
batch_size = 4
save_interval = 2000
display_interval = 500


class Pix2PixModel():
    def __init__(self, save_dir='checkpoints', log_dir='logs', gpu_ids=[0]):
        self.model_names = ['netD', 'netG']
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.gpu_ids = gpu_ids
        self.logger = Logger(log_dir)
        # Decide which device we want to run on
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=128, norm_layer=nn.BatchNorm2d, use_dropout=True).to(self.device)
        init_net(self.netG, 'normal', 0.002, [0])

        self.netD = NLayerDiscriminator(input_nc=6, ndf=128, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=True).to(self.device)
        init_net(self.netD, 'normal', 0.002, [0])

        self.fake_AB_pool = ImagePool(pool_size)

        self.criterionGAN = GANLoss(use_lsgan=not no_lsgan).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()

        # initialize optimizers
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

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
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1

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

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def visualize(self, epoc, i_batch, step):
        print('Saving samples....{}'.format(step))
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
        # ax[0].imshow(self.real_A[0].cpu().numpy().transpose(1, 2, 0))
        # ax[1].imshow(self.real_B[0].cpu().numpy().transpose(1, 2, 0))
        # ax[2].imshow(self.fake_B[0].detach().cpu().numpy().transpose(1, 2, 0))
        # plt.savefig('trash/sample_{}_{}.png'.format(epoc, i_batch))

        self.logger.scalar_summary('loss_D_fake', self.loss_D_fake, step)
        self.logger.scalar_summary('loss_D_real', self.loss_D_real, step)
        self.logger.scalar_summary('loss_D', self.loss_D, step)
        self.logger.scalar_summary('loss_G_GAN', self.loss_G_GAN, step)
        self.logger.scalar_summary('loss_G_L1', self.loss_G_L1, step)
        self.logger.scalar_summary('loss_G', self.loss_G, step)

        img = np.concatenate([
            self.real_A[0].cpu().numpy().transpose(1, 2, 0),
            self.real_B[0].cpu().numpy().transpose(1, 2, 0),
            self.fake_B[0].detach().cpu().numpy().transpose(1, 2, 0)
        ], 1)
        self.logger.image_summary('step_{}'.format(step), [img], step)

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{}_{}.pth'.format(epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                self.update_link(save_path, os.path.join(self.save_dir, 'latest_{}.pth'.format(name)))

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
                load_path = os.path.join(self.save_dir, load_filename)
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


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Lambda(distort),
        # transforms.Lambda(toGray),
    ])
    dataloader = get_folder_loader(dataroot=image_dir, transform=tf, batch_size=batch_size, shuffle=True, num_workers=2)
    model = Pix2PixModel()
    model.load_networks('latest')
    step = 186000+13000
    for epoc in range(epocs):
        for i_batch, (data, labels) in enumerate(dataloader):
            model.set_input(data)
            model.forward()
            model.backward()

            if step % display_interval == 0:
                model.visualize(epoc, i_batch, step)

            if step % save_interval == 0:
                model.save_networks(epoc)
            step += batch_size







