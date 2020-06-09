import math
import numpy as np
import torch
import torch.nn.functional as F

from third_party.matlab_imresize.imresize import imresize as matlab_imresize


def imresize(I, scalar_scale=None, output_shape=None):
  I = matlab_imresize(I.astype(np.float64), scalar_scale, output_shape)
  I = np.around(np.clip(I, 0, 255)).astype(np.uint8)
  return I


def image2patches(x, patch_size, padding=0, stride=32):
  b, c, h, w = x.shape
  h2 = math.floor((h + padding * 2 - patch_size) / stride) + 1
  w2 = math.floor((w + padding * 2 - patch_size) / stride) + 1
  crop_h = h + padding * 2 - ((h2 - 1) * stride + patch_size)
  crop_w = w + padding * 2 - ((w2 - 1) * stride + patch_size)
  if padding:
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')

  x1 = x[:, :, crop_h // 2:h + padding * 2 - crop_h // 2, crop_w // 2:w +
         padding * 2 - crop_w // 2]
  x1 = F.unfold(x1, patch_size, stride=stride)
  x1 = x1.permute(0, 2, 1).reshape(b, -1, c, patch_size, patch_size)

  x2 = x[:, :, :patch_size, :w + padding * 2 - crop_w]
  x2 = F.unfold(x2, patch_size, stride=stride)
  x2 = x2.permute(0, 2, 1).reshape(b, -1, c, patch_size, patch_size)

  x3 = x[:, :, :h + padding * 2 - crop_h, -patch_size:]
  x3 = F.unfold(x3, patch_size, stride=stride)
  x3 = x3.permute(0, 2, 1).reshape(b, -1, c, patch_size, patch_size)

  x4 = x[:, :, -patch_size:, crop_w:]
  x4 = F.unfold(x4, patch_size, stride=stride)
  x4 = x4.permute(0, 2, 1).reshape(b, -1, c, patch_size, patch_size)

  x5 = x[:, :, crop_h:, :patch_size]
  x5 = F.unfold(x5, patch_size, stride=stride)
  x5 = x5.permute(0, 2, 1).reshape(b, -1, c, patch_size, patch_size)

  x = torch.cat((x1, x2, x3, x4, x5), 1).view(-1, c, patch_size, patch_size)
  return x


def patches2image(x, image_shape, padding=0, stride=32):
  b, c, h, w = image_shape
  patch_size = x.shape[-1]
  h2 = math.floor((h + padding * 2 - patch_size) / stride) + 1
  w2 = math.floor((w + padding * 2 - patch_size) / stride) + 1
  crop_h = h + padding * 2 - ((h2 - 1) * stride + patch_size)
  crop_w = w + padding * 2 - ((w2 - 1) * stride + patch_size)
  if padding:
    x = x[..., padding:-padding, padding:-padding]
  x = x.view(b, -1, *x.shape[1:])
  out_x = torch.zeros(image_shape, dtype=x.dtype).to(x.device)
  out_o = torch.zeros_like(out_x)

  x1 = x[:, :h2 * w2]
  o1 = torch.ones_like(x1)
  x1 = x1.reshape(b, h2 * w2, -1).permute(0, 2, 1)
  o1 = o1.reshape(b, h2 * w2, -1).permute(0, 2, 1)
  x1 = F.fold(
      x1, (h - crop_h, w - crop_w), patch_size - padding * 2, stride=stride)
  o1 = F.fold(
      o1, (h - crop_h, w - crop_w), patch_size - padding * 2, stride=stride)
  out_x[:, :, crop_h // 2:h - (crop_h - crop_h // 2), crop_w // 2:w -
        (crop_w - crop_w // 2)] += x1
  out_o[:, :, crop_h // 2:h - (crop_h - crop_h // 2), crop_w // 2:w -
        (crop_w - crop_w // 2)] += o1

  weight = 0.3

  x2 = x[:, h2 * w2:h2 * w2 + w2]
  o2 = torch.ones_like(x2)
  x2 = x2.reshape(b, w2, -1).permute(0, 2, 1)
  o2 = o2.reshape(b, w2, -1).permute(0, 2, 1)
  x2 = F.fold(
      x2, (patch_size - padding * 2, w - crop_w),
      patch_size - padding * 2,
      stride=stride)
  o2 = F.fold(
      o2, (patch_size - padding * 2, w - crop_w),
      patch_size - padding * 2,
      stride=stride)
  out_x[:, :, :patch_size - padding * 2, :w - crop_w] += x2 * weight
  out_o[:, :, :patch_size - padding * 2, :w - crop_w] += o2 * weight

  x3 = x[:, h2 * w2 + w2:h2 * w2 + w2 + h2]
  o3 = torch.ones_like(x3)
  x3 = x3.reshape(b, h2, -1).permute(0, 2, 1)
  o3 = o3.reshape(b, h2, -1).permute(0, 2, 1)
  x3 = F.fold(
      x3, (h - crop_h, patch_size - padding * 2),
      patch_size - padding * 2,
      stride=stride)
  o3 = F.fold(
      o3, (h - crop_h, patch_size - padding * 2),
      patch_size - padding * 2,
      stride=stride)
  out_x[:, :, :h - crop_h, padding * 2 - patch_size:] += x3 * weight
  out_o[:, :, :h - crop_h, padding * 2 - patch_size:] += o3 * weight

  x4 = x[:, h2 * w2 + w2 + h2:h2 * w2 + w2 + h2 + w2]
  o4 = torch.ones_like(x4)
  x4 = x4.reshape(b, w2, -1).permute(0, 2, 1)
  o4 = o4.reshape(b, w2, -1).permute(0, 2, 1)
  x4 = F.fold(
      x4, (patch_size - padding * 2, w - crop_w),
      patch_size - padding * 2,
      stride=stride)
  o4 = F.fold(
      o4, (patch_size - padding * 2, w - crop_w),
      patch_size - padding * 2,
      stride=stride)
  out_x[:, :, padding * 2 - patch_size:, crop_w:] += x4 * weight
  out_o[:, :, padding * 2 - patch_size:, crop_w:] += o4 * weight

  x5 = x[:, h2 * w2 + w2 + h2 + w2:h2 * w2 + w2 + h2 + w2 + h2]
  o5 = torch.ones_like(x5)
  x5 = x5.reshape(b, h2, -1).permute(0, 2, 1)
  o5 = o5.reshape(b, h2, -1).permute(0, 2, 1)
  x5 = F.fold(
      x5, (h - crop_h, patch_size - padding * 2),
      patch_size - padding * 2,
      stride=stride)
  o5 = F.fold(
      o5, (h - crop_h, patch_size - padding * 2),
      patch_size - padding * 2,
      stride=stride)
  out_x[:, :, crop_h:, :patch_size - padding * 2] += x5 * weight
  out_o[:, :, crop_h:, :patch_size - padding * 2] += o5 * weight

  x = out_x / out_o
  return x
