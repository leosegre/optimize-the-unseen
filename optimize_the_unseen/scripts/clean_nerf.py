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
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
import os
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig


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
    # lambda for opacities loss
    opacities_lambda: float = 0.000001
    # lambda for gsplat loss
    gsplat_lambda: float = 1.0



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
        train_config.optimizers["opacities"]["scheduler"] = ExponentialDecaySchedulerConfig(
                lr_final=1.6e-8,
                max_steps=20000,
            )
        train_config.optimizers["opacities"]["optimizer"] = AdamOptimizerConfig(lr=0.001, eps=1e-15)
        # import ipdb; ipdb.set_trace()
        trainer = train_config.setup()
        trainer.setup()
        trainer.optimizers = trainer.setup_optimizers()

        load_dir = trainer.config.load_dir
        load_step = trainer.config.load_step
        if load_step is None:
            print("Loading latest Nerfstudio checkpoint from load_dir...")
            # NOTE: this is specific to the checkpoint name format
            load_step = sorted(int(x[x.find("-") + 1: x.find(".")]) for x in os.listdir(load_dir))[-1]
        load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
        assert load_path.exists(), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        trainer.optimizers.load_optimizers(loaded_state["optimizers"])
        # trainer.optimizers.optimizers["opacities"].param_groups[0]["lr"] = 0.0001
        if "schedulers" in loaded_state and trainer.config.load_scheduler:
            trainer.optimizers.load_schedulers(loaded_state["schedulers"])
        trainer.grad_scaler.load_state_dict(loaded_state["scalers"])


        start_time = time.time()

        train_pipeline = trainer.pipeline

        self.device = train_pipeline.device
        self.aabb = train_pipeline.model.scene_box.aabb.to(self.device)
        self.density_loss = MSELoss()

        train_pipeline.train()

        # add a scheduler to the optimizer of opacities
        # import ipdb; ipdb.set_trace()
        # trainer.optimizers.schedulers["opacities"] = (ExponentialDecaySchedulerConfig(
        #         lr_final=1.6e-6,
        #         max_steps=30000,
        #     ).setup().get_scheduler(optimizer=trainer.optimizers.optimizers["opacities"], lr_init=1.6e-2))
        # print(trainer.optimizers.schedulers)


        original_optimizers = trainer.optimizers.optimizers
        trainer.callbacks = train_pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=trainer.optimizers, grad_scaler=trainer.grad_scaler, pipeline=trainer.pipeline, trainer=trainer
            )
        )


        # print(trainer.optimizers.schedulers["opacities"].get_last_lr())
        # print(trainer.optimizers.schedulers["means"].get_last_lr())
        cpu_or_cuda_str: str = trainer.device.split(":")[0]

        list_num_gauss = []
        list_opacities_loss = []
        list_gsplat_loss = []

        for step in range(self.iters):
            print("step", step)

            # training callbacks before the training iteration
            for callback in trainer.callbacks:
                callback.run_callback_at_location(
                    step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                )

            needs_zero = [group for group in trainer.optimizers.parameters.keys() if step % trainer.gradient_accumulation_steps[group] == 0]
            trainer.optimizers.zero_grad_some(needs_zero)
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=trainer.mixed_precision):
                _, loss_dict, metrics_dict = trainer.pipeline.get_train_loss_dict(step=step)
                splat_loss = functools.reduce(torch.add, loss_dict.values())
                # # choose random indices of 0.5 of the gaussians
                # weights = torch.ones_like(train_pipeline.model.gauss_params["opacities"]).squeeze()
                # weights /= weights.sum()
                # random_indices = torch.multinomial(weights, int(weights.shape[0] * 0.1), replacement=False)
                opacities_loss = train_pipeline.model.gauss_params["opacities"].mean()
                loss = self.opacities_lambda * opacities_loss + self.gsplat_lambda * splat_loss
                # loss = self.opacities_lambda * opacities_loss
                print("loss", loss.item())
                print("opacities_loss", opacities_loss.item())
                # print(train_pipeline.model.gauss_params["opacities"])
                # print("opacities lr", trainer.optimizers.schedulers["opacities"].get_last_lr())
                # print("means lr", trainer.optimizers.schedulers["means"].get_last_lr())
            trainer.grad_scaler.scale(loss).backward()  # type: ignore
            needs_step = [
                group
                for group in trainer.optimizers.parameters.keys()
                if step % trainer.gradient_accumulation_steps[group] == trainer.gradient_accumulation_steps[group] - 1
            ]
            # import ipdb; ipdb.set_trace()
            trainer.optimizers.optimizer_scaler_step_some(trainer.grad_scaler, needs_step)

            scale = trainer.grad_scaler.get_scale()
            trainer.grad_scaler.update()
            # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
            if scale <= trainer.grad_scaler.get_scale():
                trainer.optimizers.scheduler_step_all(step)


            # import ipdb; ipdb.set_trace()
            # training callbacks after the training iteration
            for callback in trainer.callbacks:
                callback.run_callback_at_location(
                    step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                )

            num_gauss = train_pipeline.model.gauss_params["opacities"].shape[0]

            # store num_gauss, opacities loss and gsplat loss for each iteration, later use it to plot.
            list_num_gauss.append(num_gauss)
            list_opacities_loss.append(opacities_loss.item())
            list_gsplat_loss.append(splat_loss.item())

        # plot the opacities loss and gsplat loss and num_gauss
        import matplotlib.pyplot as plt
        plt.plot(list_num_gauss)
        plt.xlabel("iteration")
        plt.ylabel("num_gauss")
        plt.show()

        plt.plot(list_opacities_loss)
        plt.xlabel("iteration")
        plt.ylabel("opacities_loss")
        plt.show()

        plt.plot(list_gsplat_loss)
        plt.xlabel("iteration")
        plt.ylabel("gsplat_loss")
        plt.show()


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
