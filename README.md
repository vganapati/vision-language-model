# vision-language-model
Vision Language Model (VLM) running on NERSC

[Video](https://www.youtube.com/watch?v=vAmKB7iPkWw&ab_channel=UmarJamil)
[GitHub](https://github.com/hkproj/pytorch-paligemma/tree/main)
Paligemma weights: https://www.kaggle.com/models/google/paligemma-2

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
- https://huggingface.co/blog/paligemma
- https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma
- https://huggingface.co/google/paligemma-3b-pt-224
- Paligemma paper: https://arxiv.org/abs/2407.07726