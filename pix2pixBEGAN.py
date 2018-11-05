import argparse
import shutil
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.transforms import transforms
from dataset import get_folder_loader
from image_pool import ImagePool
from imageutils import distort, todict
from logger import Logger
from network import UnetGenerator, UnetDescriminator, init_net

from datetime import datetime
import os
import numpy as np
import pickle


beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
no_lsgan = False
epocs = 2000
pool_size = 50
image_size = 256

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='128_crop',  help='path to training dataset')
parser.add_argument('--valroot',  help='path to validation dataset')
parser.add_argument('--rundir', default='runs',  help='path to logs and models')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--imsize', type=int, help='size of image', default=256)
parser.add_argument('--ngf', type=int, help='generator filter size', default=128)
parser.add_argument('--ndf', type=int, help='descriminator filter size', default=64)
parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
parser.add_argument('--lr', type=float, help='learning rate', default=0.002)
parser.add_argument('--loginterval', type=int, help='learning rate', default=250)
parser.add_argument('--lrdecayInterval', type=int, help='learning rate', default=5000)
parser.add_argument('--comment', help='comments', default='')

opt = parser.parse_args()


class BEGANModel():
    def __init__(self, opt, gpu_ids=[0], continue_run=None):
        self.opt = opt
        self.kt = 0
        self.lamk = 0.001
        self.lambdaImg = 100
        self.lambdaGan = 1.0
        self.model_names = ['netD', 'netG']
        self.gpu_ids = gpu_ids

        if not continue_run:
            expname = '-'.join(['b_' + str(self.opt.batchSize), 'ngf_' + str(self.opt.ngf), 'ndf_' + str(self.opt.ndf), 'gm_' + str(self.opt.gamma)])
            self.rundir = self.opt.rundir+'/pix2pixBEGAN-'+datetime.now().strftime('%B%d-%H-%M-%S')+expname+self.opt.comment
            if not os.path.isdir(self.rundir):
                os.mkdir(self.rundir)
            with open(self.rundir+'/options.pkl', 'wb') as file:
                pickle.dump(opt, file)
        else:
            self.rundir = continue_run
            if os.path.isfile(self.rundir+'/options.pkl'):
                with open(self.rundir+'/options.pkl', 'rb') as file:
                    tmp = opt.rundir
                    tmp_lr = opt.lr
                    self.opt = pickle.load(file)
                    self.opt.rundir = tmp
                    self.opt.lr = tmp_lr

        self.netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, ngf=self.opt.ngf, norm_layer=nn.BatchNorm2d, use_dropout=True)
        self.netD = UnetDescriminator(input_nc=3, output_nc=3, num_downs=7, ngf=self.opt.ndf, norm_layer=nn.BatchNorm2d, use_dropout=True)

        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        init_net(self.netG, 'normal', 0.002, [0])
        init_net(self.netD, 'normal', 0.002, [0])

        self.netG.to(self.device)
        self.netD.to(self.device)
        self.imagePool = ImagePool(pool_size)

        self.criterionL1 = torch.nn.L1Loss()

        if continue_run:
            self.load_networks('latest')

        self.writer = Logger(self.rundir)
        self.start_step, self.opt.lr = self.writer.get_latest('misc/lr', self.opt.lr)

        # initialize optimizers
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(beta1, 0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(beta1, 0.999))


    def set_input(self, data):
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimD.zero_grad()
        fake = self.imagePool.query(self.fake_B.detach())

        recon_real_B = self.netD(self.real_B)
        recon_fake = self.netD(fake)

        d_real = torch.mean(torch.abs(recon_real_B - self.real_B))
        d_fake = torch.mean(torch.abs(recon_fake - fake))

        L_D = d_real - self.kt * d_fake
        L_D.backward()
        self.optimD.step()

        self.L_D_val = L_D.item()
        self.d_fake_cpu = d_fake.detach().cpu().item()
        self.d_real_cpu = d_real.detach().cpu().item()
        self.recon_real_B_cpu = recon_real_B.detach().cpu()
        self.recon_fake_cpu = recon_fake.detach().cpu()
        self.fake_cpu = fake.detach().cpu()

    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimG.zero_grad()

        L_Img = self.lambdaImg * self.criterionL1(self.fake_B, self.real_B)
        L_Img.backward(retain_graph=True)

        recon_fake_B = self.netD(self.fake_B)
        self.L_G_fake = self.lambdaGan * torch.mean(torch.abs(recon_fake_B - self.fake_B))
        if self.lambdaGan > 0:
            self.L_G_fake.backward()

        self.optimG.step()

        self.L_Img_cpu = L_Img.detach().cpu()
        self.L_G_fake_cpu = self.L_G_fake.detach().cpu()

    def update_K(self):
        balance = self.opt.gamma * self.d_real_cpu - self.d_fake_cpu
        self.kt = min(max(self.kt + self.lamk * balance, 0), 1)
        self.M_global = self.d_real_cpu + np.abs(balance)

    def updatelr(self):
        self.opt.lr = self.opt.lr / 2
        for param_group in self.optimD.param_groups:
            param_group['lr'] = self.opt.lr  # param_group['lr']/2
        for param_group in self.optimG.param_groups:
            param_group['lr'] = self.opt.lr  # param_group['lr']/2

    def log(self, epoch, batchn, n_iter):
        print('Writing summaries....')
        self.writer.scalar_summary('misc/M_global', self.M_global, n_iter)
        self.writer.scalar_summary('misc/kt', self.kt, n_iter)
        self.writer.scalar_summary('misc/lr', self.opt.lr, n_iter)
        self.writer.scalar_summary('loss/L_D', self.L_D_val, n_iter)
        self.writer.scalar_summary('loss/d_real', self.d_real_cpu, n_iter)
        self.writer.scalar_summary('loss/d_fake', self.d_fake_cpu, n_iter)
        self.writer.scalar_summary('loss/L_G', self.L_G_fake_cpu, n_iter)
        self.writer.scalar_summary('loss/L1', self.L_Img_cpu, n_iter)

        test_A = self.test_data['A']
        test_B = self.test_data['B']

        val_A = self.val_data['A']

        with torch.no_grad():
            fake_test_B = self.netG(test_A.to(self.device))
            fake_val_B = self.netG(val_A.to(self.device))


        images = torch.cat([test_A, test_B, fake_test_B.cpu()])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=4)
        self.writer.image_summary('Test/Fixed', [x], n_iter)

        images = torch.cat([self.real_A.detach().cpu(), self.real_B.cpu(), self.fake_B.detach().cpu()])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=4)
        self.writer.image_summary('Test/Last', [x], n_iter)

        images = torch.cat([val_A, fake_val_B.cpu()])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=4)
        self.writer.image_summary('Test/Validation', [x], n_iter)

        images = torch.cat([self.real_B.cpu(), self.recon_real_B_cpu])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=4)
        self.writer.image_summary('Discriminator/Recon_Real', [x], n_iter)

        images = torch.cat([self.fake_cpu, self.recon_fake_cpu])
        x = vutils.make_grid(images / 2 + 0.5, normalize=True, scale_each=True, nrow=4)
        self.writer.image_summary('Discriminator/Recon_Fake', [x], n_iter)

        self.save_networks(epoch)
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

    def set_val_input(self, val_data):
        self.val_data = val_data

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
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)


if __name__ == '__main__':
    print('Given opts:')
    print(opt)
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.Lambda(todict),
        ])
        valloader = get_folder_loader(dataroot=opt.valroot, transform=val_tf, batch_size=opt.batchSize, shuffle=True, num_workers=1)
    else:
        valloader = dataloader

    model = BEGANModel(opt=opt)
    opt = model.opt
    print('Loaded opts:')
    print(opt)

    test_data, _ = next(iter(dataloader))
    model.set_test_input(test_data)
    nb_batches = len(dataloader)
    n_iter = model.start_step+1
    for epoch in range(epocs):
        for i, (data, labels) in enumerate(dataloader):
            model.set_input(data)

            model.forward()
            model.backward_D()
            model.backward_G()
            model.update_K()

            LD_LG = model.L_D_val - model.d_fake_cpu
            print('Step: {}, Epoch ({}: {}/{}) M_global: {}, L_D_val: {}, d_real: {}, d_fake: {}, kt: {}, LD_LG: {}'
                  .format(n_iter, epoch, i, nb_batches, model.M_global, model.L_D_val, model.d_real_cpu, model.d_fake_cpu, model.kt, LD_LG))

            if n_iter >= opt.loginterval or n_iter >= opt.lrdecayInterval:
                if n_iter % opt.loginterval == 0:
                    test_data, _ = next(iter(valloader))
                    model.set_val_input(test_data)
                    model.log(epoch, i, n_iter)

                if n_iter % opt.lrdecayInterval == 0:
                    print('Adjusting learning rate...')
                    model.updatelr()
            n_iter += 1







