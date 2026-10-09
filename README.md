# 🔥 Source-free multitarget gradient calibration (SFMGC)

The implementation of the paper **Source-free multitarget gradient calibration: a new distribution adaptation method for fault diagnosis of rotating machinery**[PAPER_LINK].

> 
> After publication, `PAPER_LINK` will be replaced with the official paper/DOI link.

## Updating!

[NEWS!] The code and documentation of **SFMGC** are being continuously updated.

## Brief introduction

In practical industrial scenarios, source-free domain adaptation (SFDA) is an important technique to handle distribution shift and satisfy data-privacy requirements for fault diagnosis. However, existing SFDA approaches are mostly limited to single-target-domain scenarios and ignore gradient conflicts among multiple target domains. To this end, a novel source-free multitarget gradient calibration (SFMGC) framework is proposed. Firstly, label-smoothing regularization is adopted in source pretraining to mitigate over-fitting and obtain well-generalized source model initialization. Secondly, a multi-target gradient constraint (MTGC) strategy is designed to enforce gradient-direction consistency across different target domains, suppressing gradient conflict and facilitating domain-invariant feature learning. Subsequently, an inter-domain gradient alignment (IDGA) algorithm is developed, which efficiently approximates MTGC via first-order optimization and avoids the heavy computational burden brought by second-order Hessian computation. Finally, experimental results on planetary gearbox dataset and HUST bearing dataset demonstrate that SFMGC achieves state-of-the-art diagnostic performance under multiple source-free cross-working-condition transfer diagnosis tasks.

## Highlights

- A novel **source-free multi-target diagnostic paradigm SFMGC** is proposed, which can adapt to multiple heterogeneous unlabeled target domains simultaneously without accessing source-domain raw data.
- An innovative **multi-target gradient constraint (MTGC)** strategy is constructed. It constrains gradient direction consistency among different target domains from the optimization perspective, alleviating gradient conflict and negative-transfer risk in multi-target scenarios.
- A lightweight **inter-domain gradient alignment (IDGA)** algorithm is developed. It realizes the approximation of MTGC with pure first-order optimization, avoiding expensive Hessian-matrix calculation and reducing GPU memory as well as time overhead.
- Entropy minimization, diversity regularization and high-confidence pseudo-label learning are combined to form self-supervised optimization objectives for target-domain adaptation.
- Comprehensive validation is carried out on planetary gearbox and public HUST bearing datasets. SFMGC shows excellent accuracy and stability under various cross-condition transfer diagnosis tasks.

## Paper

**Source-free multitarget gradient calibration: a new distribution adaptation method for fault diagnosis of rotating machinery**
Zhiwen Ana, Quan Qiana,*, Yi Qinb, Yang Yua, Sheng Xiangc, Yuhua Chenga

a School of Automation Engineering, University of Electronic Science and Technology of China, Chengdu 611731, China
b State Key Laboratory of Mechanical Transmission, College of Mechanical and Vehicle Engineering, Chongqing University, Chongqing 400044, China
c College of Automation, Chongqing University of Posts and Telecommunications, Chongqing 400065, China

* Corresponding author: Quan Qian

**Paper link:** [To be updated](PAPER_LINK)

## SFMGC

## Code structure

```
SFMGC/
├── README.md
├── SFMGC.py             # main training & evaluation script
├── gear_dataset.py      # dataset class and data loading function
├── models_adabn_multi.py# backbone network: Multi-domain BatchNorm & Gear1DCNN_MultiDomain
└── data/
    ├── 20Hz_x_dataset_3072.csv
    ├── 20Hz_x_label_3072.csv
    ├── 25Hz_x_dataset_3072.csv
    ├── 25Hz_x_label_3072.csv
    ├── 30Hz_x_dataset_3072.csv
    └── 30Hz_x_label_3072.csv
```

## Quick start

1. Prepare your dataset and place csv files under `./data/`.
2. Modify hyper-parameters in `SFMGC.py` if needed.
3. Run:

```
python SFMGC.py
```

## Citation

If you find this work useful in your research, please consider citing our paper.

```
@article{sfmgc2026,
  title={Source-free multitarget gradient calibration: a new distribution adaptation method for fault diagnosis of rotating machinery},
  author={An, Zhiwen and Qian, Quan and Qin, Yi and Yu, Yang and Xiang, Sheng and Cheng, Yuhua},
  journal={},
  year={2026}
}
```

## Requirements

```
numpy
pandas
torch>=1.12
scipy
tqdm
```

> 
> Note: This repository is for academic research only. Commercial usage is prohibited without permission.
