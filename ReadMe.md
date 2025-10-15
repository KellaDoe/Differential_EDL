# Differential Evidential Deep Learning for Robust Meidcal Out-of-Distribution Detection

Out-of-Distribution detection method.



## Data preparation

The ISIC2019 dataset can be download at [https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019)

Please change your own path for ISIC2019 dataset in **Data/isic.py**

## Code

**visualize_isic.ipynb**

To evaluate ID classification and OOD detection:


**train.py**

To train the proposed D-EDL on existing dataset

```
python train.py --dataset isic --method rolenet
```

**dataset.py**

Load dataset, split ood categories, split data.

**Methods/ROLENet.py**

Implementation for proposed ROLENet.



The evaluation is in line with the work  "Out-of-Distribution Detection for Long-tailed and Fine-grained Skin Lesion Images" (https://arxiv.org/abs/2206.15186) in MICCAI 2022.


