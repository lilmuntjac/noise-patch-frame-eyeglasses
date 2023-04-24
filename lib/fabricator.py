from dataclasses import dataclass
import numpy as np
from pathlib import Path
from PIL import Image
import dlib

import torch
import torch.nn.functional as F
from torchvision import transforms

class NoiseOverlay:
    """
    """

    def __init__(self, batch_size=128, width=224, budget=0.1, device='cuda'):
        self.batch_size = batch_size
        self.width = width
        self.budget = budget
        self.device = device

    def apply(self, image, label, component):
        """
        add the noise onto the image, and keep the image in the valid range of value
            Input:
                image: a batch of image in shape (N, C, H, W)
                label: a batch of label in shape (N, X)
                component: an adverserial element applys to the image in shape (1, C, H, W)
            Output:
                image: a batch of image in shape (M, C, H, W)
                label: a batch of label in shape (M, X), unmodified
                no dropping of any data-label pair, so M == N
        """
        new_image = image + component
        new_image = torch.clamp(new_image, min=0.0, max=1.0)
        return new_image, label

    def clip_by_budget(self, component):
        """clip the adversarial element by budget"""
        component.data.clamp_(-self.budget, self.budget)


class HeuristicMasking:
    """
    """

    def __init__(self, mask_type, batch_size=128, width=224, sharpness=40., thickness=0.25, device='cuda'):
        self.batch_size = batch_size
        self.width = width
        self.device = device
        self.mask_type = mask_type
        match mask_type:
            case 'patch':
                self.sharpness = sharpness
                self.mask = self.get_circle_mask(sharpness)
            case 'frame':
                self.mask = self.get_frame_mask(thickness)
            case 'eyeglasses':
                self.mask = self.get_eyeglasses_mask()
                self.face_detector = dlib.cnn_face_detection_model_v1('./dlib_models/mmod_human_face_detector.dat')
                self.shape_predictor = dlib.shape_predictor('./dlib_models/shape_predictor_68_face_landmarks.dat')
                self.coord_ref = self.get_coord_reference()
            case _:
                assert False, f'the mask type {mask_type} not supported'
        self.mask = self.mask.to(device)
        self.set_identity_transform()

    # All possible masking method
    def get_circle_mask(self, sharpness):
        """Create circle mask"""
        diameter = self.width
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
    
    def get_frame_mask(self, thickness):
        """
        Create frame mask
            thickness: thickness of the frame compare to the half of the image width, in (0, 1)
        """
        base = torch.zeros(self.width, self.width)
        gap = thickness*self.width/2
        for i in range(self.width):
            for j in range(self.width):
                if i < gap or j < gap:
                    base[i][j] = 1
                if self.width-i < gap or self.width-j < gap:
                    base[i][j] = 1
        mask = base.unsqueeze(0)
        mask = torch.cat((mask, mask, mask), 0)
        mask = mask.unsqueeze(0)
        # mask is in shape (1, C, H, W)
        return mask
    
    def get_eyeglasses_mask(self, path='./eyeglasses_mask_6percent.png'):
        image = Image.open(path)
        PIL_to_tensor = transforms.ToTensor()
        mask = PIL_to_tensor(image)
        mask = mask.unsqueeze(0)
        # mask is in shape (1, C, H, W)
        return mask
    
    def get_landmark(self, image, label):
        """
        Filter batch of image to keep the landmark-detectable images. Then get their landmark.
            Input:
                image: a batch of image from dataloader
                label: a batch of label for the image loaded
            Output:
                image: filtered images that we can get the landmark from it
                label: filtered images' corresponding labels
                landmark: landmark for the filtered images (list of objects)
        """
        rgb_image = image.clone().detach().cpu().permute(0, 2, 3, 1).numpy()*255.9999
        rgb_image = rgb_image.astype(np.uint8)
        detectable, landmark = list(), list()
        for idx in range(self.batch_size):
            detected_face = self.face_detector(rgb_image[idx], 1)
            if len(detected_face) != 1:
                continue # only 1 face is allowed
            landmark.append(self.shape_predictor(rgb_image[idx], detected_face[0].rect))
            detectable.append(idx)
        filtered_image = image[detectable, :, :, :]
        filtered_label = label[detectable,]
        return filtered_image, filtered_label, landmark

    def get_coord_reference(self):
        theta = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
        grid = F.affine_grid(theta, (1, 3, self.width, self.width), align_corners=False)
        coord_ref =grid[0,0,:,0]
        return coord_ref
    
    def get_torch_coord(self, point_list):
        """
        Get the torch coordinate for a list of points, integer only
            Input:
                point_list, list of points, a point is define as a list of 2 elements as x and y coordinate.
                coordinate is count by pixel, the origin is at the top left.
        """
        new_coord = list()
        for point in point_list:
            x, y = int(point[0]), int(point[1])
            # landmark forced to be within the image
            x, y = max(0, min(x, 223)), max(0, min(y, 223))
            new_x, new_y = self.coord_ref[x], self.coord_ref[y]
            new_coord.append([new_x, new_y])
        return new_coord

    # All kind of transforms
    def set_identity_transform(self):
        """
        Get identity transformation matrix for batch of circles
        The circle will be place at the centor of the image, with the diameter same as the image width
        """
        theta = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
        theta = theta.repeat(self.batch_size, 1, 1)
        self.theta = theta.to(self.device)

    def set_random_transform(self, rotation, ratio_2_image, avoid_from_center=None):
        """
        Get random transformation matrix for batch of circles
        Put the circles in all possible, non-clipping locations
            Input:
                rotation: a positive float, rotate in both CW and CCW, in radian
                ratio_2_image: 2-tuple as (ratio_min, ratio_max), ratio of the circle to the whole image (0,1]
                    ratio is counted in area, not the width, 1 will make the circle be cliped in some area
                avoid_from_center: a float, distance from the centor of the image (0,1] not to cover
                    default is 'None' and the circle can be placed anywhere non-cliped
            Output:
                transformation matrix theta in shape (N, 2, 3)
        """ 
        theta = torch.empty(0)
        # create one transformation matrix at a time
        for b in range(self.batch_size):
            # rotation and scaling
            rot = (-2*rotation)*torch.rand(1) + rotation
            rot_matrix = torch.tensor(
                [[torch.cos(-rot), -torch.sin(-rot)],
                 [torch.sin(-rot), torch.cos(-rot)]]
            )
            scale = map(lambda x : 2*np.sqrt(x/np.pi), ratio_2_image)
            scale_min, scale_max = scale
            scale = (scale_min-scale_max)*torch.rand(1) + scale_max
            inv_scale = 1.0 / scale
            scale_matrix = torch.tensor(
                [[inv_scale, 0],
                 [0, inv_scale]]
            )
            xform_matrix = torch.mm(rot_matrix, scale_matrix)
            # translation
            if avoid_from_center != None:
                range_min, range_max = avoid_from_center+scale, 1-scale
                if range_min >= range_max:
                    print(f'range min: {range_min}, range max: {range_max}')
                    assert False, f'Patch is too large (or too close) to avoid the center of the image.'
                # keep trying until it fit
                while True:
                    rnd_min, rnd_max = -(1-scale), 1-scale
                    shift_x, shift_y = (rnd_min-rnd_max)*torch.rand(2) + rnd_max
                    if shift_x >= range_min or shift_y >= range_min:
                        break
                shift_x, shift_y = shift_x*inv_scale, shift_y*inv_scale
            else:
                if scale <= 1.0:
                    shift_min, shift_max = -(1-scale)/scale, (1-scale)/scale
                else:
                    shift_min, shift_max = 0.0, 0.0
                shift_x, shift_y = (shift_min-shift_max)*torch.rand(2) + shift_max
            xform_matrix = torch.cat((xform_matrix, torch.tensor([[shift_x], [shift_y]])), 1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        self.theta = theta.to(self.device)

    def set_eyeglasses_transform(self, landmark, reference=[[73,75],[149,75],[111,130]]):
        """
        Get transformation matrix for batch of eyeglasses
        Use the landmark from the face to transform the eyeglasses
            Input:
                landmark: a list of landmark objevt return by dlib shape predictor
                reference: the startng point coordinate for the eyeglasses mask. 
                    They're 3 points: left eye, right eye, and noise tip
                    coordinate is count by pixel, the origin is at the top left.
            Output:
                transformation matrix theta in shape (N, 2, 3)
        """

        reference = self.get_torch_coord(reference)
        theta = torch.empty(0)
        for lm in landmark:
            # get the transformed points from the landmark
            left_eye, right_eye, noise_tip = (lm.part(36)+lm.part(39))/2 ,(lm.part(42)+lm.part(45))/2, lm.part(33)
            destination = [[left_eye.x, left_eye.y], [right_eye.x, right_eye.y], [noise_tip.x, noise_tip.y]]
            destination = self.get_torch_coord(destination)
            for point in destination:
                point.append(1)
            destination = torch.tensor(destination, dtype=torch.float)
            outset = torch.tensor(reference, dtype=torch.float)
            xform_matrix = torch.linalg.solve(destination, outset).transpose(0,1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        self.theta = theta.to(self.device)

    def apply(self, image, label, component):
        """
        Apply the adversarial component onto the image.
            Input:
                image: a batch of image in shape (N, C, H, W)
                label: a batch of label in shape (N, X)
                component: an adverserial element be masked then apply to the image in shape (1, C, H, W)
            Output:
                image: a batch of image in shape (M, C, H, W)
                label: a batch of label in shape (M, X), unmodified
                no dropping of any data-label pair, so M == N
        """
        mask = self.mask.repeat(image.shape[0], 1, 1, 1)
        component = component.repeat(image.shape[0], 1, 1, 1)
        grid = F.affine_grid(self.theta, image.shape, align_corners=False)
        xform_mask = F.grid_sample(mask, grid, align_corners=False)
        xform_component = F.grid_sample(component, grid, mode='bilinear', align_corners=False)
        inv_mask = 1 - xform_mask
        return image*inv_mask + xform_component*xform_mask, label
    
    def to_valid_image(self, component):
        """
        clip the adversarial element back to valid image range
            Patch must be a valid image
        """
        component.data.clamp_(0.0, 1.0)
