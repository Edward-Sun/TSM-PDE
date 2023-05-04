# TSM-PDE

See ["A Neural PDE Solver with Temporal Stencil Modeling"](https://arxiv.org/abs/2302.08105) for the paper associated with this codebase.

Parts of the codebase is adapted from [google/jax-cfd](https://github.com/google/jax-cfd).

<p align="center">
  <img src="https://github.com/Edward-Sun/TSM-PDE/blob/main/tsm_main.png?raw=true" alt="TSM Illustration"/>
</p>

## Setup

```bash
conda env create -f environment.yml
conda activate cfd
cd jax-cfd
pip install jaxlib
pip install -e ".[complete]"
cd ..
```

## Data

Both the training and evaluation data can be deterministically generated.
Please see the [data_generation](data_generation.md) for more details.

## Reproduction

Please check the [reproducing_scripts](reproducing_scripts.md) for more details.

## Pretrained Checkpoints

Please download the pretrained model checkpoints from [here](https://drive.google.com/drive/folders/19dVHw8G8-0RCK2Y7p-oGlG2lzfKJwQ8S?usp=share_link).

## Reference

If you found this codebase useful, please consider citing the following papers:

Temporal Stencil Modeling:

```
@article{sun2023tsm,
  title={A Neural PDE Solver with Temporal Stencil Modeling},
  author={Sun, Zhiqing and Yang, Yiming and Yoo, Shinjae},
  journal={arXiv preprint arXiv:2302.08105},
  year={2023}
}
```

Learned Interpolation:

```
@article{Kochkov2021-ML-CFD,
  author = {Kochkov, Dmitrii and Smith, Jamie A. and Alieva, Ayya and Wang, Qing and Brenner, Michael P. and Hoyer, Stephan},
  title = {Machine learning{\textendash}accelerated computational fluid dynamics},
  volume = {118},
  number = {21},
  elocation-id = {e2101784118},
  year = {2021},
  doi = {10.1073/pnas.2101784118},
  publisher = {National Academy of Sciences},
  issn = {0027-8424},
  URL = {https://www.pnas.org/content/118/21/e2101784118},
  eprint = {https://www.pnas.org/content/118/21/e2101784118.full.pdf},
  journal = {Proceedings of the National Academy of Sciences}
}
```
