import inspect
import matplotlib.pyplot as plt
import torchvision.transforms as tf
import torch.nn as nn
import threading
from contextlib import contextmanager
from timeit import default_timer
import torchvision.utils as vutils
import PIL
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class YCbCr2RGB(nn.Module):
    def __init__(self):
        super(YCbCr2RGB, self).__init__()

    def forward(self, y, cb, cr):
        R = y + 1.402*(cr-128/255)
        G = y - 0.344136*(cb-128/255) - 0.714136*(cr-128/255)
        B = y + 1.772*(cb-128/255)
        return torch.cat([R,G,B], 1)

def makeParallel(model):
    if 'device_ids' in dir(model): #already parallel
        model = model.module
    return nn.DataParallel(model).cuda()


def savefig(x, name=None, color='g'):
    if not name:
        s =  inspect.stack()[1] # caller's stack
        for k, v in list(s.frame.f_locals.items()):
            if v is x:
                name = k
    #plt.clf()
    fig = plt.figure(figsize=(64, 10))
    plt.plot(x, color)
    plt.savefig(name+'.png')    
    #plt.savefig('psnr{}.png'.format(epoch))
    plt.close(fig)

def imshow(x, nrow=8, title='default'):
    if len(x.size())==4:
        if type(x) is torch.autograd.variable.Variable:
            x = x.data.cpu()            
        else:
            x = x.cpu()
    x = vutils.make_grid(x, normalize=True, scale_each=True)    
    x = tf.ToPILImage()(x)
    x.show(title=title)

def log_variable(*args):
    s =  list(inspect.stack()[1].frame.f_locals.items()) # caller's stack
    resstr = ''
    for x in args:
        for k, v in s:
            if v is x:
                resstr += k+': {:.2E}'.format(x)+'\t'
                break
    print(resstr)

import numpy as np
import torch
from PIL import Image
def resize_4d(im, newsize):
    #input: 4d image in NCHW
    sc = tf.Scale(newsize)
    n,c,h,w = im.size()
    newtensor = torch.Tensor(n, c, newsize, newsize)
    for i in range(n):
        for j in range(c):
            newtensor[i] = tf.ToTensor()(sc(tf.ToPILImage()(im[i])))
    return newtensor


def gradStrength(model):
     return [m.grad.abs().mean().data[0] for m in model.parameters() if m.grad ]
    #    if m.grad:
    #        print(m.grad.abs().mean().data[0])
#    gradGstrength = netG.module.conv2._parameters['weight'].grad.abs().mean().data[0]
