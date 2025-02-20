# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reference implementation of AugMix's data augmentation method in numpy."""
from .augs import augmentations
import numpy as np
from PIL import Image
import torch

# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)


def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  res = np.asarray(pil_img) / 255.
  return res


def augment_and_mix_pil(image: Image, severity=3, width=3, depth=-1, alpha=1.):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))

  mix = np.zeros_like(image)
#   print(mix.shape, image.shape)
  for i in range(width):
    image_aug = image.copy()
    d = depth if depth > 0 else np.random.randint(1, 4)
    for _ in range(d):
      op = np.random.choice(augmentations)
      image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    # print(mix.shape, image_aug.shape)
    mix = mix + ws[i] * normalize(image_aug)

  mixed = (1 - m) * normalize(image) + m * mix
  return mixed


# def augment_and_mix_tensor(image: torch.Tensor, severity=3, width=3, depth=-1, alpha=1.):
#   """Perform AugMix augmentations and compute mixture.
#   Args:
#     image: Raw input image as float32 np.ndarray of shape (h, w, c)
#     severity: Severity of underlying augmentation operators (between 1 to 10).
#     width: Width of augmentation chain
#     depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
#       from [1, 3]
#     alpha: Probability coefficient for Beta and Dirichlet distributions.
#   Returns:
#     mixed: Augmented and mixed image.
#   """
#   image = Image.fromarray(image.cpu().numpy())
#   image = augment_and_mix_pil(image, severity, width, depth, alpha)
#   return torch.from_numpy(image)


def augment_and_mix_tensors(images: torch.Tensor, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    assert images.device == torch.device('cpu')
    
    res = []
    for image in images:
        gray_img = False
        if image.size(0) == 1:
            gray_img = True
            image = torch.cat([image, image, image])
        image = image.numpy().transpose(1, 2, 0)
        
        aug_image = augment_and_mix_pil(image, severity, width, depth, alpha)
        if gray_img:
            aug_image = aug_image.transpose(2, 0, 1)[0: 1]
        res += [torch.from_numpy(aug_image).float()]
    return torch.stack(res)
