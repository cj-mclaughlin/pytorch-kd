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

## Baseline Models (without KD)
Results are almost identical to that of [CRD](https://github.com/HobbitLong/RepDistiller). Slight differences may come as a result of using pre-activation ResNets.
WRN40-2 - 75.34
WRN40-1 - 71.81
WRN28-4 - 78.31
WRN16-4 - 76.60
WRN16-2 - 72.39
ResNet56 - 72.50
ResNet20 - 69.27

## KD Results
TODO.