# pytorch-kd
This repo aims to reproduce a wide range of works in literature in knowledge distillation.

## Target Papers

### Knowledge Distillation
(In Progress)
- KD: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- FitNets: [FitNets: Hints for Thin Deep nets](https://arxiv.org/abs/1412.6550)
- AT: [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/abs/1612.03928)
- DML: [Deep Mutual Learning](https://arxiv.org/abs/1706.00384)
- BYOT: [Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://arxiv.org/abs/1905.08094)
- CRD: [Contrastive Representation Distillation](http://arxiv.org/abs/1910.10699)

### Robustness/Augmentation
- SaliencyMix: [SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization](https://arxiv.org/abs/2006.01791)

## Training Details
This repo follows settings of [CRD](https://github.com/HobbitLong/RepDistiller), namely we train for 240 epochs with initial learning rate of 0.05, batch size 64, weight decay 5e-4, and momentum of 0.9. Learning rate is decayed by a factor of 10 at epochs 150, 180, and 210. 

## Download Models
All trained models can be found [here](https://drive.google.com/drive/folders/1gL8ensehP_JTkgNf2RNRkoDC2SXAznU-?usp=sharing).

## Baseline Models

Wide Residual Networks
|                     | WRN40-2 | WRN40-1 | WRN28-4 | WRN16-4 | WRN16-2 | WRN16-1 |
|---------------------|---------|---------|---------|---------|---------|---------|
| Baseline            |  76.44  |  71.91  |  79.03  |  77.35  |  74.03  |  67.85  |

ResNet (pre-activation)
|                     | ResNet56 | ResNet20 | ResNet8 |
|---------------------|----------|----------|---------|
| Baseline            |  72.55   |  69.51   |  61.11  |

## Results on CIFAR100C
Models may also be evaluated on the [CIFAR100-Corrupted benchmark](https://arxiv.org/abs/1903.12261) to evaluate the robustness of each training method. Below shows the performance of WRN40-2 on clean and corrupted CIFAR with varying training schemes.

| WRN40-2             | CIFAR100 | CIFAR100C |
|---------------------|----------|-----------|
| Baseline            |  76.44   |  48.12    |
| SaliencyMix         |  78.26   |  49.06    |
!
## KD Results
TODO.
