#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal

def get_rays_torch(focal, c2w, H=64,W=64):
    """Computes rays using a General Pinhole Camera Model
    Assumes self.h, self.w, self.focal, and self.cam_to_world exist
    """
    x, y = torch.meshgrid(
        torch.arange(W),  # X-Axis (columns)
        torch.arange(H),  # Y-Axis (rows)
        indexing='xy')
    camera_directions = torch.stack(
        [(x - W * 0.5 + 0.5) / focal,
            -(y - H * 0.5 + 0.5) / focal,
            -torch.ones_like(x)],
        dim=-1).to(c2w)

    # Rotate ray directions from camera frame to the world frame
    directions = ((camera_directions[ None,..., None, :] * c2w[None,None, None, :3, :3]).sum(axis=-1))  # Translate camera frame's origin to the world frame
    origins = torch.broadcast_to(c2w[ None,None, None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, axis=-1, keepdims=True)

    return torch.cat((origins,viewdirs),dim=-1)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class RCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, uid, delta_polar, delta_azimuth, delta_radius, opt,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", SSAA=False
                 ):
        super(RCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.delta_polar = delta_polar
        self.delta_azimuth = delta_azimuth
        self.delta_radius = delta_radius
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01
        
        if SSAA:
            ssaa = opt.SSAA
        else:
            ssaa = 1

        self.image_width = opt.image_w * ssaa
        self.image_height = opt.image_h * ssaa

        self.trans = trans
        self.scale = scale

        RT = torch.tensor(getWorld2View2(R, T, trans, scale))
        self.world_view_transform = RT.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # self.rays = get_rays_torch(fov2focal(FoVx, 64), RT).cuda()
        self.rays = get_rays_torch(fov2focal(FoVx, self.image_width//8), RT, H=self.image_height//8, W=self.image_width//8).cuda()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

