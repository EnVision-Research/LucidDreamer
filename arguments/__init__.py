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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group
    
    def load_yaml(self, opts=None):
        if opts is None:
            return
        else:
            for key, value in opts.items():
                try:
                    setattr(self, key, value)
                except:
                    raise Exception(f'Unknown attribute {key}')

class GuidanceParams(ParamGroup):
    def __init__(self, parser, opts=None):
        self.guidance = "SD"        
        self.g_device = "cuda"

        self.model_key = None
        self.is_safe_tensor = False
        self.base_model_key = None

        self.controlnet_model_key = None

        self.perpneg =  True
        self.negative_w = -2.
        self.front_decay_factor = 2.
        self.side_decay_factor = 10.   
        
        self.vram_O = False
        self.fp16 = True
        self.hf_key = None
        self.t_range = [0.02, 0.5]     
        self.max_t_range = 0.98
        
        self.scheduler_type = 'DDIM'
        self.num_train_timesteps = None 

        self.sds = False
        self.fix_noise = False
        self.noise_seed = 0

        self.ddim_inv = False
        self.delta_t = 80
        self.delta_t_start = 100
        self.annealing_intervals = True
        self.text = ''
        self.inverse_text = ''
        self.textual_inversion_path = None
        self.LoRA_path = None
        self.controlnet_ratio = 0.5
        self.negative = ""
        self.guidance_scale = 7.5
        self.denoise_guidance_scale = 1.0
        self.lambda_guidance = 1.

        self.xs_delta_t = 200
        self.xs_inv_steps = 5
        self.xs_eta = 0.0

        # multi-batch
        self.C_batch_size = 1

        self.vis_interval = 100

        super().__init__(parser, "Guidance Model Parameters")


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False, opts=None):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self.pretrained_model_path = None
        self._images = "images"
        self.workspace = "debug"
        self.batch = 10  
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = False
        self.opt_path = None
        
        # augmentation
        self.sh_deg_aug_ratio = 0.1
        self.bg_aug_ratio = 0.5
        self.shs_aug_ratio = 0.0
        self.scale_aug_ratio = 1.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser, opts=None):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser, opts=None):
        self.iterations = 5000# 10_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0050
        self.feature_lr_final = 0.0030

        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001


        self.geo_iter = 0
        self.as_latent_ratio = 0.2
        # dense

        self.resnet_lr = 1e-4
        self.resnet_lr_init = 2e-3
        self.resnet_lr_final = 5e-5


        self.scaling_lr_final = 0.001
        self.rotation_lr_final = 0.0002

        self.percent_dense = 0.003
        self.densify_grad_threshold = 0.00075

        self.lambda_tv = 1.0 # 0.1
        self.lambda_bin = 10.0
        self.lambda_scale = 1.0
        self.lambda_sat = 1.0
        self.lambda_radius = 1.0
        self.densification_interval = 100
        self.opacity_reset_interval = 300
        self.densify_from_iter = 100
        self.densify_until_iter = 30_00 
        
        self.use_control_net_iter = 10000000 
        self.warmup_iter = 1500 
        
        self.use_progressive = False
        self.save_process = True
        self.pro_frames_num = 600
        self.pro_render_45 = False
        self.progressive_view_iter = 500
        self.progressive_view_init_ratio = 0.2

        self.scale_up_cameras_iter = 500
        self.scale_up_factor = 0.95
        self.fovy_scale_up_factor = [0.75, 1.1]
        self.phi_scale_up_factor = 1.5
        super().__init__(parser, "Optimization Parameters")


class GenerateCamParams(ParamGroup):
    def __init__(self, parser):
        self.init_shape = 'sphere'
        self.init_prompt = ''      
        self.use_pointe_rgb  = False
        self.radius_range = [5.2, 5.5] #[3.8, 4.5] #[3.0, 3.5]
        self.max_radius_range = [3.5, 5.0]
        self.default_radius = 3.5
        self.theta_range = [45, 105]
        self.max_theta_range = [45, 105]
        self.phi_range = [-180, 180]
        self.max_phi_range = [-180, 180]
        self.fovy_range = [0.32, 0.60] #[0.3, 1.5] #[0.5, 0.8]  #[10, 30]
        self.max_fovy_range = [0.16, 0.60]
        self.rand_cam_gamma = 1.0
        self.angle_overhead = 30
        self.angle_front =60
        self.render_45 = True
        self.uniform_sphere_rate = 0
        self.image_w = 512
        self.image_h = 512 # 512
        self.SSAA = 1
        self.init_num_pts = 100_000
        self.default_polar = 90
        self.default_azimuth = 0
        self.default_fovy = 0.55 #20
        self.jitter_pose = True
        self.jitter_center = 0.05
        self.jitter_target = 0.05
        self.jitter_up = 0.01
        self.device = "cuda"
        super().__init__(parser, "Generate Cameras Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
