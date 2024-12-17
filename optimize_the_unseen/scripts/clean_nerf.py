# !/usr/bin/env python
"""
vf.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import tyro
import torch
import numpy as np
import nerfstudio
import nerfacc
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.model_components.losses import MSELoss
import yaml
import functools
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation

@dataclass
class CleanNeRF:
    """Load a checkpoint, compute VF, and save it to a npy file."""

    # Path to config YAML file.
    load_config: Path
    # Step to load.
    load_step: int = None
    # number of iterations on the trainset
    iters: int = 1000
    # Number of samples per step
    samples_per_step: int = 131072
    # seed
    seed: int = 42



    def get_density_from_pipeline(self, pipeline, random_points, normalize=True):
        positions = random_points
        if normalize:
            # Compute the density field at the random points
            if pipeline.model.field.spatial_distortion is not None:
                positions = pipeline.model.field.spatial_distortion(positions)
                positions = (positions + 2.0) / 4.0
            else:
                positions = SceneBox.get_normalized_positions(random_points, self.aabb)
            # Make sure the tcnn gets inputs between 0 and 1.
            selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
            positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        h = pipeline.model.field.mlp_base(positions_flat).view(*positions.shape[:-1], -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, pipeline.model.field.geo_feat_dim], dim=-1)
        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        # density = density_before_activation.to(positions)
        if normalize:
            density = density * selector[..., None]
            base_mlp_out = base_mlp_out * selector[..., None]
        return density, base_mlp_out


    def main(self) -> None:
        """Main function."""

        # seed everything
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        CONSOLE.log(f"Loading pre-set config from: {self.load_config}")
        train_config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)
        train_config.load_dir = train_config.get_checkpoint_dir()
        train_config.load_step = self.load_step
        train_config.save_only_latest_checkpoint = False
        train_config.vis = None
        trainer = train_config.setup()
        trainer.setup()

        start_time = time.time()

        train_pipeline = trainer.pipeline

        self.device = train_pipeline.device
        self.aabb = train_pipeline.model.scene_box.aabb.to(self.device)
        self.density_loss = MSELoss()

        train_pipeline.train()

        original_optimizers = trainer.optimizers.optimizers
        trainer.callbacks = train_pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=trainer.optimizers, grad_scaler=trainer.grad_scaler, pipeline=trainer.pipeline, trainer=trainer
            )
        )

        for step in range(self.iters):
            print("step", step)

            # Assuming self.aabb is a tensor of shape (2, 3) where the first row is the min corner
            # and the second row is the max corner of the AABB.
            min_corner = self.aabb[0]
            max_corner = self.aabb[1]

            # Generate random points within the AABB
            random_points = torch.rand((self.samples_per_step, 3), device=self.aabb.device) * (
                        max_corner - min_corner) + min_corner

            # Compute the density field at the random points
            train_density, train_rgb_embeddings = self.get_density_from_pipeline(train_pipeline, random_points, normalize=False)
            train_proposal_density = []
            # compute the density field at the random points for proposal networks
            for i_level in range(train_pipeline.model.proposal_sampler.num_proposal_network_iterations):
                train_proposal_density.append(train_pipeline.model.density_fns[i_level](random_points))
            train_proposal_density = torch.stack(train_proposal_density)

            density_mask = torch.zeros_like(train_density)
            proposal_density_mask = torch.zeros_like(train_proposal_density)

            clean_density = density_mask.detach()
            clean_proposal_density = proposal_density_mask.detach()
            train_density = torch.sigmoid(train_density)
            clean_density = torch.sigmoid(clean_density).detach()
            train_proposal_density = torch.sigmoid(train_proposal_density)
            clean_proposal_density = torch.sigmoid(clean_proposal_density).detach()


            cpu_or_cuda_str: str = trainer.device.split(":")[0]

            density_lambda = 0.1
            nerf_lambda = 1.0
            proposal_density_lamba = 0.01

            proposal_density_loss = self.density_loss(train_proposal_density, clean_proposal_density)
            density_loss = self.density_loss(train_density, clean_density)

            _, loss_dict, metrics_dict = trainer.pipeline.get_train_loss_dict(step=trainer._start_step + step)
            nerf_loss = functools.reduce(torch.add, loss_dict.values())

            loss = density_lambda * density_loss + nerf_lambda * nerf_loss \
                        + proposal_density_lamba * proposal_density_loss
            print("loss", loss.item())

            with (torch.autocast(device_type=cpu_or_cuda_str, enabled=trainer.mixed_precision)):
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    trainer.optimizers.zero_grad_all()
                    trainer.grad_scaler.scale(loss).backward()
                    trainer.optimizers.optimizer_scaler_step_all(trainer.grad_scaler)
                    trainer.grad_scaler.update()
                    trainer.optimizers.scheduler_step_all(trainer._start_step + step)
                else:
                    print("backward skipped")

        end_time = time.time()
        print("Done")
        trainer.optimizers.optimizers = original_optimizers

        trainer.config.timestamp = "clean"
        trainer.config.load_dir = train_config.get_checkpoint_dir()
        trainer.checkpoint_dir = train_config.get_checkpoint_dir()
        trainer.config.pipeline.model.background_color = "white"

        trainer.config.save_config()
        trainer.save_checkpoint(trainer._start_step + self.iters - 1)
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")



def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(CleanNeRF).main()


if __name__ == "__main__":
    entrypoint()
