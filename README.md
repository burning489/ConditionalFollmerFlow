## Official Implementations of `Deep Conditional Distribution Learning via  Conditional Föllmer Flow`

We present the implementations of our proposed conditional Föllmer flow in this repository, and examine its performance on 2 simulation studies and 3 real data analyses. 
We also compare the conditional Föllmer flow with existing NNKCDE, FlexCode, conditional GAN, conditional VAE, conditional VE-SDE and conditional Trigonometric flow. 
All the results in our manuscript can be reproduced with the `slurm.sh` in each sub-folder.

### Dependencies

- `pip install -r requirements` 
- 
- You need to manual install `nnkcde` from `https://github.com/lee-group-cmu/NNKCDE` as its authors do not offer an official installation through `PyPi` or `conda`.

### Hardware specification

We carry out the numerical experiments on 3 nodes of a NVIDIA 4xV100 cluster. 
In general, a machine with a 16GB NVIDIA GPU would satisfy the requirements for reproducing.

If you are using a personal desktop computer, you need to remove the `slurm` specificaitons in the `slurm.sh` and specify your python environments. 

If you are using a slurm cluster, you might need to modify the `slurm` specifications such as the name of `partition`.
