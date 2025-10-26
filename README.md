# Optimize the Unseen - Fast NeRF Cleanup with Free Space Prior
## NeurIPS 2025
### [Project Page](https://leosegre.github.io/optimize-the-unseen/) | [Paper](https://arxiv.org/abs/2412.12772)

By [Leo Segre](https://scholar.google.co.il/citations?hl=iw&user=A7FWhoIAAAAJ) and [Shai Avidan](https://scholar.google.co.il/citations?hl=iw&user=hpItE1QAAAAJ)

This repo is the official implementation of "[Optimize the Unseen - Fast NeRF Cleanup with Free Space Prior](https://arxiv.org/abs/2412.12772)".

https://github.com/user-attachments/assets/2a192776-624d-449f-b516-69d8621a68f0

## Citation
If you find this useful, please cite this work as follows:
```bibtex
@misc{segre2024optimizeunseenfast,
      title={Optimize the Unseen -- Fast NeRF Cleanup with Free Space Prior}, 
      author={Leo Segre and Shai Avidan},
      year={2024},
      eprint={2412.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.12772}, 
}
```

### About
Optimize the Unseen is a method for fast cleanup NeRF from artifacts and floaters by optimizing the NeRF on regions unsen by the original set of training cameras.

### Installation
Optimize the Unseen is built on top of [Nerfstudio](https://docs.nerf.studio/).
After cloning Optimize the Unseen repository, install Nerfstudio as a package by following the installation guide on [Nerfstudio installation page](https://docs.nerf.studio/quickstart/installation.html)

Specifically, perform the following steps
1. [Create environment](https://docs.nerf.studio/quickstart/installation.html#create-environment)
2. [Dependencies](https://docs.nerf.studio/quickstart/installation.html#dependencies)
3. [Installing nerfstudio](https://docs.nerf.studio/quickstart/installation.html#installing-nerfstudio). Follow the **From pip** guidelines, no need to clone Nerfstudio.


Then install Optimize the Unseen as a package using `pip install -e .`
This will allow you to run `ns-clean-nerf` command in the terminal.

### Running
First train a Nerfacto model using the `ns-train` command:
```
ns-train nerfacto  --vis viewer --data {PATH_TO_DATA} --experiment-name {EXP_NAME} --output-dir {OUTPUT_DIR} --timestamp base  --relative-model-dir=nerfstudio_models  --max-num-iterations=30000  nerfstudio-data --downscale-factor 1
```

Then clean the pre-trained NeRF using `ns-clean-nerf` command:
```
ns-clean-nerf --load-config {PATH_TO_ORIGINAL_CONFIG}
```

The new cleaned NeRF model is now stored in the same directory of the original one under the timestamp "clean".

### View the cleaned model
Use 'ns-viewer' to view the new cleaned model.
```
ns-viewer --load-config {PATH_TO_NEW_CONFIG}
```

You can use any nerfstudio command on the new model (ns-render, ns-export etc).


### Built On
<a href="https://github.com/nerfstudio-project/nerfstudio">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.nerf.studio/_images/logo.png" />
<!-- /pypi-strip -->
    <img alt="nerfstudio logo" src="https://docs.nerf.studio/_images/logo.png" width="150px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- A collaboration friendly studio for NeRFs

