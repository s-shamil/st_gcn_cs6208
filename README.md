# ST-GCN
A PyTorch implementation of
S. Yan, Y. Xiong, and D. Lin, “**Spatial temporal graph convolutional networks for skeleton-based action recognition**,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

This repository contains a PyTorch implementation of ST-GCN developed as part of coursework in CS6208: CS6208 Advanced Topics in Artificial Intelligence.

## Model details
Our implemented network consists of only the components that were described in the original paper with the mentioned hyperparameters. Due to time constraints, we skip the large Kinetics dataset and conduct experiments on NTU RGB+D X-sub (cross-subject) dataset. Additionally, we also adopt the method for hand action recognition and apply on hand pose data found from [Assembly101](https://assembly-101.github.io/).

Our code supports `Uni-labeling` and `Distance Partitioning` as neighborhood partition methods. The paper shows detailed results for Kinetics dataset, where `Distance Partitioning` performs competitively with their best strategy `Spatial Configuration`. However, `Distance Partitioning` is simpler while the motivation for designing `Spatial Configuration` does not hold for hand poses. Therefore, here we show the results with `Distance Partitioning` only.

## Experiments
Validation accuracy reproduced for **NTU RGB+D X-sub dataset** is **73.60%** (`Distance Partitioning`). This is compared to the reported accuracy of 81.5% (`Spatial Configuration`). There are two possible reasons for the performance difference: i) Although Kinetics exhibit similar performance for these two partitions, it might impact NTU RGB+D. The authors do not provide detailed performance results for NTU RGB+D. ii) Their original imnplementation may have additional layers and hyperparameters that are not mentioned in the paper for brevity.

On **Assembly101**, our validation accuracy is **23.99%**.

## Training and Testing

We collect the preprocessed dataset as instructed [here](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).

To train on NTU RGB+D dataset

```bash
python train.py
```

To train on Assembly101 dataset

```bash
python train_assembly.py
```

Inside training code the data directory and model names can be adjusted.

```python
data_dir = '/home/salman/Datasets/data/Assembly101/'
model_details = "distancePartition_stgcn_EdgeImp_handAction"

phase = 'train' # 'test': load best model and run on val set, 
               # 'train': train model
```
