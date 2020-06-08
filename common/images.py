import numpy as np
from third_party.matlab_imresize.imresize import imresize as matlab_imresize


def imresize(I, scalar_scale=None, output_shape=None):
  I = matlab_imresize(I.astype(np.float64), scalar_scale, output_shape)
  I = np.around(np.clip(I, 0, 255)).astype(np.uint8)
  return I
