# Multi-Points PolarMask: One-Stage Instance Segmentation with Polar Respresentations

## Architecture and Training 
![image-20190807160835333](imgs/pipeline.png)


**Train:**
- ```python tools/train.py configs/polarmask/4gpu/polar_768_1x_r50.py --work_dir rescale_new_ap```

## Performances
![Graph](imgs/visual.png)
MP-Head module has multiple parallel networks, each for processing one feature map Fi, i = 2, 3, . . . , p.
Each network has the following branches: (i) one classification branch, (ii) one Polar centerness branch, (iii) one mask
regression branch, and (iv) four auxiliary-center branches.
Similar to PolarMask, the first three branches compute the matrices C ∈ RW × RH × Rk, P ∈ RW × RH ,and M ∈ RW × RH × Rn, respectively. 
Each auxiliary point branch computes a matrix Am ∈ RW × RH × R2, m ∈ {1, 2, . . . , 4}, 
where each tensor (i, j, ∗) ∈ Am is a 2D displacement vector to locate the auxiliary center corresponding to the center (i, j, ∗) ∈ C in quad-rang Qm

![Table](imgs/performance.png)

## Citations
@article{xie2019polarmask,
  title={PolarMask: Single Shot Instance Segmentation with Polar Representation},
  author={Xie, Enze and Sun, Peize and Song, Xiaoge and Wang, Wenhai and Liu, Xuebo and Liang, Ding and Shen, Chunhua and Luo, Ping},
  journal={arXiv preprint arXiv:1909.13226},
  year={2019}
}