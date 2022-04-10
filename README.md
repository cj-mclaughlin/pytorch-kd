# pytorch-kd
Knowledge Distillation Papers in Pytorch

## Papers

- KD: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- FitNets: [FitNets: Hints for Thin Deep nets](https://arxiv.org/abs/1412.6550)
- AT: [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/abs/1612.03928)
- DML: [Deep Mutual Learning](https://arxiv.org/abs/1706.00384)
- BYOT: [Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://arxiv.org/abs/1905.08094)
- CRD: [Contrastive Representation Distillation](http://arxiv.org/abs/1910.10699)

## Training Details
We follow the settings of [CRD](https://github.com/HobbitLong/RepDistiller), name we train for 240 epochs with initial learning rate of 0.05, batch size 64, weight decay 5e-4, and momentum of 0.9. Learning rate is decayed by a factor of 10 at epochs 150, 180, and 210. 

## Download Models
All trained models can be found [here](https://drive.google.com/drive/folders/1gL8ensehP_JTkgNf2RNRkoDC2SXAznU-?usp=sharing).

## Baseline Models (without KD)
Slight improvements compared to the CRD repo may come as a result of using pre-activation ResNets.

|                     | WRN40-2 | WRN40-1 | WRN28-4 | WRN16-4 | WRN16-2 | ResNet56 | ResNet20 | ResNet8 |
|---------------------|---------|---------|---------|---------|---------|----------|----------|---------|
| Baseline<br>(No KD) |  76.44  |  71.91  |  79.03  |  77.35  |  74.03  |  72.55   |  69.51   |  61.11  |

## KD Results
TODO.