# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

from xxsubtype import bench

from datetime import datetime
import json
import types
import torch
import mediapy as media
from dataclasses import dataclass
from time import time
from pathlib import Path
from typing import Optional
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
import tyro
import numpy as np
import cv2
import torchmetrics

import nerfstudio
import pkg_resources

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

from optimize_the_unseen.metrics.image_metrics import PSNRModule, SSIMModule, LPIPSModule


from PIL import Image
import matplotlib.pyplot as plt

from torchmetrics.classification import Dice
import os
from nerfstudio.engine.trainer import TrainerConfig

import numpy as np
import cv2


from nerfstudio.utils import colors


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w,
                                        self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w,
                                             self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        # setup a look-up table for spatial kernel
        LUT_s = np.exp(-0.5 * (np.arange(self.pad_w + 1) ** 2) / self.sigma_s ** 2)
        # setup a look-up table for range kernel
        LUT_r = np.exp(-0.5 * (np.arange(256) / 255) ** 2 / self.sigma_r ** 2)
        # compute the weight of range kernel by rolling the whole image
        wgt_sum, result = np.zeros(padded_img.shape), np.zeros(padded_img.shape)
        for x in range(-self.pad_w, self.pad_w + 1):
            for y in range(-self.pad_w, self.pad_w + 1):
                # method 1 (easier but slower)
                dT = LUT_r[np.abs(np.roll(padded_guidance, [y, x], axis=[0, 1]) - padded_guidance)]
                r_w = dT if dT.ndim == 2 else np.prod(dT, axis=2)  # range kernel weight
                s_w = LUT_s[np.abs(x)] * LUT_s[np.abs(y)]  # spatial kernel
                t_w = s_w * r_w
                padded_img_roll = np.roll(padded_img, [y, x], axis=[0, 1])
                for channel in range(padded_img.ndim):
                    result[:, :, channel] += padded_img_roll[:, :, channel] * t_w
                    wgt_sum[:, :, channel] += t_w
        output = (result / wgt_sum)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, :]

        return np.clip(output, 0, 255).astype(np.uint8)


def plot_errors(ratio_removed, ause_err, ause_err_by_var, err_type, scene_no, output_path): #AUSE plots, with oracle curve also visible
    plt.plot(ratio_removed, ause_err, '--')
    plt.plot(ratio_removed, ause_err_by_var, '-r')
    # plt.plot(ratio_removed, ause_err_by_var - ause_err, '-g') # uncomment for getting plots similar to the paper, without visible oracle curve
    path = output_path.parent / Path("plots") 
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path/ Path('plot_'+err_type+'_'+str(scene_no)+'.png'))
    plt.figure()

def plot_single_error(ratio_removed, err, err_type, scene_no, output_path): #AUSE plots, with oracle curve also visible
    plt.plot(ratio_removed, err, '-r')
    # plt.plot(ratio_removed, ause_err_by_var - ause_err, '-g') # uncomment for getting plots similar to the paper, without visible oracle curve
    path = output_path.parent / Path("plots")
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path/ Path('plot_'+err_type+'_'+str(scene_no)+'.png'))
    plt.figure()

def visualize_ranks(unc ,gt, colormap='jet'):
    flattened_unc = unc.flatten()
    flattened_gt = gt.flatten()
    
    # Find the ranks of the pixel values
    ranks_unc = np.argsort(np.argsort(flattened_unc)) 
    ranks_gt = np.argsort(np.argsort(flattened_gt)) 
    
    max_rank = max(np.max(ranks_unc),np.max(ranks_gt))
    
    cmap = plt.get_cmap(colormap, max_rank)
    
    # Normalize the ranks to the range [0, 1]
    normalized_ranks_unc = ranks_unc / max_rank
    normalized_ranks_gt = ranks_gt / max_rank
    
    # Apply the colormap to the normalized ranks
    colored_ranks_unc = cmap(normalized_ranks_unc)
    colored_ranks_gt = cmap(normalized_ranks_gt)
    
    colored_unc = colored_ranks_unc.reshape((*unc.shape,4))
    colored_gt = colored_ranks_gt.reshape((*gt.shape,4))
    
    return colored_unc, colored_gt

def get_filtered_image_metrics(self, 
                               outputs: Dict[str, torch.Tensor],
                               batch: Dict[str, torch.Tensor],
                               add_nb_mask=False,
                               visibility_mask: torch.Tensor=None,
                               ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    
    image = batch["image"]
    rgb = outputs["rgb"]
    acc = outputs["accumulation"]
    depth = outputs["depth"]
    h,w = rgb.shape[0], rgb.shape[1]

    # Normalize image to 0-1 if needed
    if image.max() > 1:
        image = image / 255.0
    image = torch.clip(torch.tensor(media.resize_image(image.cpu(), (h, w))).to(self.device),0.,1.)

    # Crop visibility mask to match the image size
    if visibility_mask is not None and add_nb_mask:
        if visibility_mask.shape[0] != h or visibility_mask.shape[1] != w:
            visibility_mask = visibility_mask[:h, :w]  # Crop extra rows/columns if necessary

    #implementing masked psnr,lpisp,ssim like https://github.com/ethanweber
    #/nerfbusters/blob/1f4240344ecff1313f6dfa7be5e06fe7d3e29154/scripts/launch_nerf.py#L258
    depth_mask = (depth < 2.0).float()
    acc_mask = (acc >= 0.98).float()


    if add_nb_mask:
        # if thresh == 0.0:
        #     mask = depth_mask[...,0]
        # else:
        mask = acc_mask[...,0]
        # mask = visibility_mask
        mask_to_dice = mask.int()
        # mask = mask * visibility_mask
        mask = mask[..., None].repeat(1, 1, 3)
    else:
        # if thresh == 0.0:
        #     mask = depth_mask[...,0]
        # else:
        mask = acc_mask[...,0]
        mask_to_dice = mask.int()
        # mask = mask * visibility_mask
        mask = mask[..., None].repeat(1, 1, 3)

    # print(batch)
    #
    # # Save rgb, image and mask for debugging
    # im = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype('uint8'))
    # im.save(f"renders/image.jpeg")
    # im = Image.fromarray((rgb.squeeze().cpu().numpy() * 255).astype('uint8'))
    # im.save(f"renders/rgb.jpeg")
    # im = Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype('uint8'), mode='L')
    # im.save(f"renders/mask.jpeg")

    image, rgb = image * mask, rgb * mask


    # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
    image = image.permute(2, 0, 1).unsqueeze(0)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)
    m = mask.permute(2, 0, 1).unsqueeze(0)[:, 0:1]

    # print(image.shape, rgb.shape, acc.shape)



    psnr = float(self.psnr_module(rgb, image, m)[0])
    ssim = float(self.ssim_module(rgb, image, m)[0])
    lpips = float(self.lpips_module(rgb, image, m)[0])
    dice_score = Dice().to(self.device)
    
    metrics_dict = {"psnr": float(psnr), "ssim": float(ssim)}  # type: ignore
    metrics_dict["lpips"] = float(lpips)
    if add_nb_mask:
        metrics_dict["coverage"] = float((mask[..., 0] * visibility_mask).sum() / visibility_mask.sum() * 100)
    else:    
        metrics_dict["coverage"] = float((mask[..., 0] * visibility_mask).sum()/(image.shape[-1]*image.shape[-2]) * 100)

    # Dice score - reshape the mask to be [2, H, W] for the dice score, 2 is the number of classes and represents onehot encoding
    # mask_to_dice = torch.stack([1 - mask[..., 0], mask[..., 0]]).unsqueeze(0)
    # visibility_mask_to_dice = torch.stack([1 - visibility_mask, visibility_mask]).unsqueeze(0)
    # mask_to_dice = mask[..., 0].int()
    # mask_to_dice = torch.ones_like(mask_to_dice)

    if add_nb_mask:
        visibility_mask_to_dice = visibility_mask.int()
        metrics_dict["dice"] = dice_score(mask_to_dice, visibility_mask_to_dice)
    else:
        metrics_dict["dice"] = dice_score(mask_to_dice, torch.ones_like(mask_to_dice))
    return metrics_dict

def normalize_for_colormap(x: torch.Tensor) -> torch.Tensor:
    """Normalize the input tensor for visualization with a colormap."""
    # set x[-inf] to the minimum value. If x is all -inf, set it to 0
    if torch.isinf(x).all():
        x[torch.isinf(x)] = 0
    else:
        x[torch.isneginf(x)] = x[~torch.isinf(x)].min()
    return (x - x.min()) / (x.max() - x.min())


def get_average_filtered_image_metrics(self, step: Optional[int] = None, thresh_values: Optional[torch.Tensor] = None):
    """Iterate over all the images in the eval dataset and get the average.
    From https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/pipelines/base_pipeline.py#L342

    Returns:
        metrics_dict: dictionary of metrics
    """
    self.eval()
    num_images = len(self.datamanager.fixed_indices_eval_dataloader)

    # Override evaluation function
    self.model.get_image_metrics_and_images = types.MethodType(get_filtered_image_metrics, self.model)

    self.model.psnr_module = PSNRModule().to(self.device)
    self.model.ssim_module = SSIMModule().to(self.device)
    self.model.lpips_module = LPIPSModule().to(self.device)
    
    views = ["view"]
    psnr = [["psnr"]]
    lpips = [["lpisp"]]
    ssim = [["ssim"]]
    coverage = [["coverage"]]
    dice = [["dice"]]

    if isinstance(self.datamanager.fixed_indices_eval_dataloader, list):
        is_splatfacto = True
        for camera in self.datamanager.fixed_indices_eval_dataloader:
            camera[0].rescale_output_resolution(1./self.downscale_factor)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
        view_no = 0

        for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
            if not is_splatfacto:
                camera_ray_bundle = camera_ray_bundle.generate_rays(camera_indices=0, keep_shape=True)
            if self.add_nb_mask:
                base_path = self.nb_mask_path
                scene_name = base_path.name
                # Base path and view number
                image_path_png = str(base_path) + "/{:05d}.png".format(view_no)
                image_path_jpg = str(base_path) + "/{:05d}.jpg".format(view_no)

                # Check which file exists and read it
                if os.path.exists(image_path_png):
                    pseudo_gt_visibility = media.read_image(image_path_png)
                elif os.path.exists(image_path_jpg):
                    pseudo_gt_visibility = media.read_image(image_path_jpg)
                else:
                    print(image_path_png)
                    raise FileNotFoundError("Image not found in PNG or JPG format.")

                pseudo_gt_visibility = torch.from_numpy(pseudo_gt_visibility).long().to(self.device)
                pseudo_gt_visibility = pseudo_gt_visibility.squeeze() > 0
            else:
                pseudo_gt_visibility = 1
                scene_name = self.datamanager.get_datapath().stem


            if is_splatfacto:
                with torch.no_grad():
                    outputs = self.model.get_outputs(camera_ray_bundle)
            else:
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)


            # Make sure that the directory exists, else make it
            Path(f"renders/{scene_name}").mkdir(parents=True, exist_ok=True)

            im = Image.fromarray((batch["image"].cpu().numpy()).astype('uint8'))
            im.save(f"renders/{scene_name}" / Path(str(view_no) + "-img-" + ".jpeg"))
            # im = Image.fromarray((pseudo_gt_visibility.cpu().numpy() * 255).astype('uint8'))
            # im.save(f"renders/{scene_name}" / Path(str(view_no) + "-pseudo_gt_visibility.jpeg"))
            im = Image.fromarray((outputs["rgb"].cpu().numpy() * 255).astype('uint8'))
            im.save(f"renders/{scene_name}" / Path(str(view_no) + "-rgb-" + "splatfacto_clean" + ".jpeg"))
            metrics_dict = self.model.get_image_metrics_and_images(outputs, batch, self.add_nb_mask, pseudo_gt_visibility)
            psnr[0].append(float(metrics_dict["psnr"]))
            lpips[0].append(float(metrics_dict["lpips"]))
            ssim[0].append(float(metrics_dict["ssim"]))
            coverage[0].append(float(metrics_dict["coverage"]))
            dice[0].append(float(metrics_dict["dice"]))
            print("view:", view_no, "psnr:", psnr[0][-1], "coverage:", coverage[0][-1], "dice:", dice[0][-1])
                


            views.append(str(view_no))    
            view_no +=1
            progress.advance(task)
            

    # average the metrics list
    metrics_dict = {}

    lists = [*psnr, *ssim, *lpips, *coverage, *dice]
    for l in lists:
        l.append(sum(l[1:])/view_no)
    views.append('average')
    lists = [views] + lists 
    self.train()
    return {}, lists


def get_time_of_forward_pass(self):
    """Takes the time of forward pass of the first image.
    Returns:
        Time in milliseconds
    """
    self.eval()
    num_images = len(self.datamanager.fixed_indices_eval_dataloader)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
        view_no = 0

        for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
            camera_ray_bundle = camera_ray_bundle.generate_rays(camera_indices=0, keep_shape=True)
            # time this the following line
            inner_start = time()
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            end_time = time()
            # Calculate time in milliseconds
            time_in_ms = (end_time - inner_start)*1000
            print("Time taken for forward pass in milliseconds:", time_in_ms)
            # Measure the Memory footprint
            print(torch.cuda.memory_summary(device=None, abbreviated=True))
            break

    return time_in_ms


@dataclass
class ComputeMetrics:
    """Load a checkpoint, compute some metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # dataset path
    dataset_path: Path = Path("./data")
    # downscale factor
    downscale_factor: float = 1.0
    # add nerfbusters mask
    nb_mask: bool = False
    # if nb_mask set to true, specify the path to the masks
    visibility_path: Path = Path("./data/nerfbusters/aloe/visibility_masks")
    # measure the time taken for each forward pass
    measure_time: bool = False
    # New data path
    data_path: Path = None
    # Eval on train set
    eval_on_train: bool = False


    def main(self) -> None:
        """Main function."""

        def update_config(config: TrainerConfig) -> TrainerConfig:
            # data_manager_config = config.pipeline.datamanager
            if self.data_path is not None:
                assert hasattr(config, "data")
                setattr(config, "data", self.data_path)
                assert hasattr(config.pipeline.datamanager, "data")
                setattr(config.pipeline.datamanager, "data", self.data_path)
                print("config.data",config.data)
                print("config.pipeline.datamanager.data", config.pipeline.datamanager.data)
            if self.eval_on_train:
                assert hasattr(config.pipeline.datamanager.dataparser, "eval_frame_indices")
                setattr(config.pipeline.datamanager.dataparser, "eval_frame_indices", (0,))
            return config

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config, update_config_callback=update_config)
        else:
            config, pipeline, checkpoint_path = eval_setup(self.load_config, update_config_callback=update_config)
        
        # Dynamic change of get_outputs method to include uncertainty
        self.device = pipeline.device
        pipeline.model.dataset_path = self.dataset_path
        pipeline.model.output_path = self.output_path
        pipeline.model.white_bg = True
        pipeline.model.black_bg = False
        pipeline.model.background_color = colors.WHITE

        pipeline.expname = config.experiment_name
        pipeline.add_nb_mask = self.nb_mask
        pipeline.nb_mask_path = self.visibility_path
        pipeline.downscale_factor = self.downscale_factor


        # Override evaluation function
        pipeline.get_average_eval_image_metrics = types.MethodType(get_average_filtered_image_metrics, pipeline)

        if self.measure_time:
            pipeline.get_time_of_forward_pass = types.MethodType(get_time_of_forward_pass, pipeline)
            time_in_ms = pipeline.get_time_of_forward_pass()
            benchmark_info = {
                "experiment_name": config.experiment_name,
                "method_name": config.method_name,
                "checkpoint": str(checkpoint_path),
                "time_in_ms": time_in_ms,
            }
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
            return

        assert self.output_path.suffix == ".json"
        metrics_dict, metric_lists = pipeline.get_average_eval_image_metrics()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        
            
        # Save output to output file
        nb_filter = '_nb' if self.nb_mask else ''
        timestamp = datetime.now().timestamp()
        date_time = datetime.fromtimestamp(timestamp)
        str_date_time = date_time.strftime("%d-%m-%Y-%H%M%S")
        csv_path = str(self.output_path).split('.')[0] + '_' + config.experiment_name + '_'+ str_date_time + nb_filter +'.csv'
        
        np.savetxt(csv_path, [p for p in zip(*metric_lists)], delimiter=',', fmt='%s')
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeMetrics).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs


def get_parser_fn(): return tyro.extras.get_parser(ComputeMetrics)  # noqa
