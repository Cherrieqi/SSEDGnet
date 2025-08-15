# SSEDGnet

![SSEDGnet](https://github.com/Cherrieqi/SSEDGnet/blob/main/Figures/SSEDGnet.png)


[A Shift-reduced Sample Expansion Domain Generalization Network for Hyperspectral Image Cross-Domain Classification](https://ieeexplore.ieee.org/document/11124263)



## Requirements

This code is based on **Python 3.10** and **Pytorch 1.12**.

*Installation list:*

**· pytorch**

**· matplotlib**

**· opencv-python**

**· scipy**

**· h5py**

**· tqdm**

**· scikit-learn**


## Models

**· SD--H13+H18 :** [model5.pth](https://pan.baidu.com/s/1T-JZScEGPR2415Rr5h70QQ?pwd=ifdr)

**· SD--PU+PC :** [model5.pth](https://pan.baidu.com/s/1Sn3kkk34ivW6AMkUi4U5yw?pwd=h8zw)


## Datasets

**· [raw](https://pan.baidu.com/s/1iDQoBf2sfl6WAXyOXC15FQ?pwd=9azr) :** Houston2013 / Houston2018 / PaviaU / PaviaC

**· [H13+H18--PU/PC](https://pan.baidu.com/s/1FRozdjaxXablec2JdclPUg?pwd=smy2) :** gen_H13 / gen_H18 / gen_PU / gen_PC

**· [PU+PC--H13/H18](https://pan.baidu.com/s/1g0pHClw-um-RRhWcdIDtrQ?pwd=npar) :** gen_PU / gen_PC / gen_H13 / gen_H18 




Getting start:
· Dataset structure

data/H1318
├── gen_H13
│   ├── img.npy
│   └── gt.npy
├── gen_H18
│   ├── img.npy
│   └── gt.npy
├── gen_PC
│   ├── img.npy
│   └── gt.npy
└── gen_PU
     ├── img.npy
     └── gt.npy
     
data/PUPC
├── gen_PU
│   ├── img.npy
│   └── gt.npy
├── gen_PC
│   ├── img.npy
│   └── gt.npy
├── gen_H13
│   ├── img.npy
│   └── gt.npy
└── gen_H18
     ├── img.npy
     └── gt.npy
     
data/raw
├── Houston2013
│   ├── Houston.mat
│   └── Houston_gt.mat
├── Houston2018
│   ├── HoustonU.mat
│   └── HoustonU_gt.mat
├── PaviaC
│   ├── pavia.mat
│   └── pavia_gt.mat
└── PaviaU
     ├── paviaU.mat
     └── paviaU_gt.mat


















