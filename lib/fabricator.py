from dataclasses import dataclass
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

class NoiseOverlay:
    """
    """

    def __init__(self, batch_size=128, width=224, budget=0.1, device='cuda'):
        self.batch_size = batch_size
        self.width = width
        self.budget = budget
        self.device = device

    def apply(self, image, label, master):
        """
        add the noise onto the image, and keep the image in the valid range of value
            Input:
                image: a batch of image in shape (N, C, H, W)
                label: a batch of label in shape (N, X)
                master: an adverserial element applys to the image in shape (1, C, H, W)
            Output:
                image: a batch of image in shape (M, C, H, W)
                label: a batch of label in shape (M, X), unmodified
                no dropping of any data-label pair, so M == N
        """
        new_image = image + master
        new_image = torch.clamp(new_image, min=0.0, max=1.0)
        return new_image, label

    def clip_by_budget(self, master):
        """clip the adversarial element by budget"""
        master.data.clamp_(-self.budget, self.budget)


class CircleMasking:
    """
    """

    def __init__(self, batch_size=128, width=224, sharpness=40., device='cuda'):
        self.batch_size = batch_size
        self.width = width
        self.sharpness = sharpness
        self.device = device
        self.mask = self.get_circle_mask(width, sharpness).to(device)
        self.set_identity_transform()

    def get_circle_mask(self, width, sharpness):
        """Create circle mask"""

        diameter = width
        x = torch.linspace(-1, 1, diameter)
        y = torch.linspace(-1, 1, diameter)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        z = (xx**2 + yy**2) ** sharpness
        mask = 1 - np.clip(z, -1, 1)
        mask = mask.unsqueeze(0)
        mask = torch.cat((mask, mask, mask), 0)
        mask = mask.unsqueeze(0)
        # mask is in shape (1, C, H, W)
        return mask

    def set_identity_transform(self):
        """The identity transformation matrix"""

        theta = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
        theta = theta.repeat(self.batch_size, 1, 1)
        self.theta = theta.to(self.device)

    def set_random_transform(self, rotation, ratio):
        """
        Get random transformation matrix for a circle
        Put the circle in all possible, non-clipping locations
            Input:
                rotation: 2-tuple as (rotation_min, rotation_max), in radian
                scale: 2-tuple as (ratio_min, ratio_max), ratio of the circle to the whole image (0,1]
            Output:
                transformation matrix theta in shape (N, 2, 3)
        """

        theta = torch.empty(0)
        for b in range(self.batch_size):
            rot = (-2*rotation)*torch.rand(1) + rotation
            rot_matrix = torch.tensor(
                [[torch.cos(-rot), -torch.sin(-rot)],
                 [torch.sin(-rot), torch.cos(-rot)]]
            )
            scale = map(lambda x : 2*np.sqrt(x/np.pi), ratio)
            scale_min, scale_max = scale
            scale = (scale_min-scale_max)*torch.rand(1) + scale_max
            inv_scale = 1.0 / scale
            scale_matrix = torch.tensor(
                [[inv_scale, 0],
                 [0, inv_scale]]
            )
            xform_matrix = torch.mm(rot_matrix, scale_matrix)
            # translation (bound by the room left by scaling)
            if scale <= 1.0:
                shift_min, shift_max = -(1-scale)/scale, (1-scale)/scale
            else:
                shift_min, shift_max = 0.0, 0.0
            shift_x, shift_y = (shift_min-shift_max)*torch.rand(2) + shift_max
            xform_matrix = torch.cat((xform_matrix, torch.tensor([[shift_x], [shift_y]])), 1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        self.theta = theta.to(self.device)

    def set_constrained_transform(self, rotation, ratio, dodge):
        """
        Get random transformation matrix for a circle
        Put the circle in all possible, non-clipping locations
            Input:
                rotation: 2-tuple as (rotation_min, rotation_max), in radian
                scale: 2-tuple as (ratio_min, ratio_max), ratio of the circle to the whole image (0,1]
                dodge: float, distance from the centor of the image
            Output:
                transformation matrix theta in shape (N, 2, 3)
        """
        
        theta = torch.empty(0)
        for b in range(self.batch_size):
            # rotation(in radian) & scale
            rot = (-2*rotation)*torch.rand(1) + rotation
            rot_matrix = torch.tensor(
                [[torch.cos(-rot), -torch.sin(-rot)],
                 [torch.sin(-rot), torch.cos(-rot)]]
            )
            scale = map(lambda x : 2*np.sqrt(x/np.pi), ratio)
            scale_min, scale_max = scale
            scale = (scale_min-scale_max)*torch.rand(1) + scale_max
            inv_scale = 1.0 / scale
            scale_matrix = torch.tensor(
                [[inv_scale, 0],
                 [0, inv_scale]]
            )
            xform_matrix = torch.mm(rot_matrix, scale_matrix)
            # translation (bound by the room left by scaling)
            range_min, range_max = dodge+scale, 1-scale
            if range_min >= range_max:
                print(f'range min: {range_min}, range max: {range_max}')
                assert False, f'Patch is too large (or too close) to avoid the center of the image.'
            # 
            while True:
                rnd_min, rnd_max = -(1-scale), 1-scale
                shift_x, shift_y = (rnd_min-rnd_max)*torch.rand(2) + rnd_max
                if shift_x >= range_min or shift_y >= range_min:
                    break
            shift_x, shift_y = shift_x*inv_scale, shift_y*inv_scale
            xform_matrix = torch.cat((xform_matrix, torch.tensor([[shift_x], [shift_y]])), 1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        self.theta = theta.to(self.device)

    def apply(self, image, label, master):
        """
        Stamp the circle onto the image
            Input:
                image: a batch of image in shape (N, C, H, W)
                label: a batch of label in shape (N, X)
                master: an adverserial element be masked then apply to the image in shape (1, C, H, W)
            Output:
                image: a batch of image in shape (M, C, H, W)
                label: a batch of label in shape (M, X), unmodified
                no dropping of any data-label pair, so M == N
        """

        mask = self.mask.repeat(self.batch_size, 1, 1, 1)
        master = master.repeat(self.batch_size, 1, 1, 1)
        
        grid = F.affine_grid(self.theta, image.shape, align_corners=False)
        xform_mask = F.grid_sample(mask, grid, align_corners=False)
        xform_master = F.grid_sample(master, grid, mode='bilinear', align_corners=False)
        inv_mask = 1 - xform_mask
        return image*inv_mask + xform_master*mask, label
