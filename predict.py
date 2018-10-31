import torch
import torch.nn as nn
from torchvision.transforms import transforms
from dataset import get_coco_loader, get_folder_loader, toGray
from logger import Logger
from network import UnetGenerator, NLayerDiscriminator, init_net
import numpy as np
import matplotlib.pyplot as plt
import os

image_dir = 'y:/Dataset/B_Gray'
# ann_file = '/home/alex/Datasets/coco/annotations/instances_val2017.json'

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
no_lsgan = False
image_size = 256
batch_size = 4


class Pix2PixModel():
    def __init__(self, save_dir='checkpoints', log_dir='logs', gpu_ids=[0]):
        self.model_names = ['netG']
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.gpu_ids = gpu_ids
        self.logger = Logger(log_dir)
        # Decide which device we want to run on
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        self.netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=128, norm_layer=nn.BatchNorm2d, use_dropout=True).to(self.device)
        init_net(self.netG, 'normal', 0.002, [0])

    def set_input(self, data):
        self.real_A = data.to(self.device)

    def forward(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
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


if __name__ == '__main__':
    from scipy.misc import imsave

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    dataloader = get_folder_loader(dataroot=image_dir, transform=tf, batch_size=batch_size, shuffle=False, num_workers=2)
    model = Pix2PixModel()
    model.load_networks('latest')

    step = 0
    for i_batch, (data, labels) in enumerate(dataloader):
        model.set_input(data)
        model.forward()

        image_batch = model.fake_B.cpu().numpy()
        for i in range(image_batch.shape[0]):
            img = image_batch[i].transpose(1, 2, 0)
            imsave('results/image_{}.png'.format(step), img)
            step += 1







