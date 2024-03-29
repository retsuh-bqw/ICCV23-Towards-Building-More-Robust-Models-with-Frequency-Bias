# ICCV 2023-Towards Building More Robust Models with Frequency Bias

<p align="left">
<a href="https://arxiv.org/abs/2307.09763"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
  
## Paper Abstract
The vulnerability of deep neural networks to adversarial samples has been a major impediment to their broad applications, despite their success in various fields. Recently, some works suggested that adversarially-trained models emphasize the importance of low-frequency information to achieve higher robustness. While several attempts have been made to leverage this frequency characteristic, they have all faced the issue that applying low-pass filters directly to input images leads to irreversible loss of discriminative information and poor generalizability to datasets with distinct frequency features. This paper presents a plug-and-play module called the Frequency Preference Control Module that adaptively reconfigures the low- and high-frequency components of intermediate feature representations, providing better utilization of frequency in robust learning. Empirical studies show that our proposed module can be easily incorporated into any adversarial training framework, further improving model robustness across different architectures and datasets. Additionally, experiments were conducted to examine how the frequency bias of robust models impacts the adversarial training process and its final robustness, revealing interesting insights.


## Usage
**Train a ResNet18 model on CIFAR10:**

*python train.py --data_root dataset_path --dataset CIFAR10 --weight_decay 3.5e-3 --lr 0.01 --batch_size 128 --epoch 120 --model resnet18*

**Train a WRN-34-10 model on CIFAR10:**

*python train.py --data_root dataset_path --dataset CIFAR10 --weight_decay 5e-4 --lr 0.1 --batch_size 128 --epoch 60 --model wrn-34-10*

**Test a model under PGD-50 attack:**

*python test.py --weights checkpoint_path --attack PGD --step 50 --dataset dataset_name --model model_name*


#### Note： 
One of our mian contribution is the Frequency Preference Control Module (FPCM) as proposed in the paper, which only involves relevant changes in the model definition script. You can migrate the relevant implementation to any of the other adversarial training code bases.

For example, plz refer to the repo: [Github](https://github.com/retsuh-bqw/ICCV23-DiffusionModel-AT), where we yield a promising **63.63\%** AutoAttack accuracy with 1M additional synthetic data for training. The original repo is at: [GitHub](https://github.com/wzekai99/DM-Improves-AT)

## Citation
If you find our work useful in your research, please consider citing:
````
@InProceedings{Bu_2023_ICCV,
    author    = {Bu, Qingwen and Huang, Dong and Cui, Heming},
    title     = {Towards Building More Robust Models with Frequency Bias},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4402-4411}
}
````
