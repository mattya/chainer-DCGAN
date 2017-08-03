import pickle
import numpy as np
from PIL import Image
import os
from io import BytesIO
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

nz = 100

model_file = 'generator_model.h5'
out_file = 'output.png'


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z=L.Linear(
                nz, 6 * 6 * 512, initialW=chainer.initializers.Normal(0.02 * math.sqrt(nz) / math.sqrt(10))),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=chainer.initializers.Normal(
                0.02 * math.sqrt(4 * 4 * 512) / math.sqrt(10))),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=chainer.initializers.Normal(
                0.02 * math.sqrt(4 * 4 * 256) / math.sqrt(10))),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=chainer.initializers.Normal(
                0.02 * math.sqrt(4 * 4 * 128) / math.sqrt(10))),
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=chainer.initializers.Normal(
                0.02 * math.sqrt(4 * 4 * 64) / math.sqrt(10))),
            bn0l=L.BatchNormalization(6 * 6 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64)
        )

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z))),
                      (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = self.dc4(h)
        return x


def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))


xp = numpy

gen = Generator()

serializers.load_hdf5(model_file, gen)


pylab.rcParams['figure.figsize'] = (22.0, 22.0)
pylab.clf()
vissize = 100
z = (xp.random.uniform(-1, 1, (100, 100)).astype(np.float32))
z = Variable(z)
with chainer.using_config('test', True):
    x = gen(z)
x = x.data
for i_ in range(100):
    tmp = ((np.vectorize(clip_img)(x[i_, :, :, :]) + 1) / 2).transpose(1, 2, 0)
    pylab.subplot(10, 10, i_ + 1)
    pylab.imshow(tmp)
    pylab.axis('off')
pylab.savefig(out_file)
