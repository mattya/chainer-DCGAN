import pickle
import numpy as np
from PIL import Image
import os
from StringIO import StringIO
import math
import pylab


import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L


import numpy

import settings

nz = settings.CONFIG['nz']
cfg = settings.CONFIG['visualize']

model_file = cfg['paths']['model_file']
out_file = cfg['paths']['out_file']

out_height_num = cfg['out_height_num']
out_width_num = cfg['out_width_num']
out_height = cfg['out_height']
out_width = cfg['out_width']
vissize = out_height_num * out_width_num

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 6*6*512, wscale=0.02*math.sqrt(nz)),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0l = L.BatchNormalization(6*6*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x



def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))


xp = numpy

gen = Generator()

serializers.load_hdf5(model_file, gen)


pylab.rcParams['figure.figsize'] = (out_width, out_height)
pylab.clf()
z = (xp.random.uniform(-1, 1, (vissize, nz)).astype(np.float32))
z = Variable(z)
x = gen(z, test=True)
x = x.data
for i_ in range(vissize):
    tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
    pylab.subplot(out_height_num, out_width_num, i_+1)
    pylab.imshow(tmp)
    pylab.axis('off')
pylab.savefig(out_file)
