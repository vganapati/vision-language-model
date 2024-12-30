# vision-language-model
Vision Language Model (VLM) running on NERSC


## Setup instructions

module load python
module avail pytorch
module load pytorch/2.3.1

conda create -n vlm
conda activate vlm


cd vision-language-model

## References


- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
-[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)