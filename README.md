## Table of contents
- [1 介绍](#1-介绍)
- [2 代码和安装](#2-代码和安装)
  * [2.1 安装依赖](#21-安装依赖)
  * [2.2 准备环境](#22-准备环境)
- [3 运行代码](#3-运行代码)
  * [3.1 设置](#31-设置)
  * [3.2 离线准备](#32-离线准备)
  * [3.3 在线训练](#33-在线训练)
  * [3.4 实验结果](#34-实验结果)
- [4 架构与实现](#4-架构与实现)

![](https://github.com/LINC-BIT/BlockTrainer/blob/master/docs/arch.png)

## 1 介绍

边缘侧大模型外部环境的不确定性（如路边摄像头画面中天气、光照、物体密度的变化），导致其输入数据分布持续改变，因此需进行重训以维持高精度.受限于设备可用资源和重训窗口，现有技术仅能训练固定压缩模型，其有限的泛化能力导致模型精度显著降低.本文提出云边协同大模型块粒度重训方法，引入模型重训缩放定律评估不同块对边缘侧当前数据的精度贡献，以此为依据生成有限资源下最优重训方案，将云平台大模型中精度最相关部分动态转换为边缘侧可重训小模型，构建大小模型协同训练系统.通过真实云边平台上对比实验表明本文方法可以在相同资源消耗下提升大模型重训精度81.24%，并支持最大至330亿参数大模型重训.

## 2 代码和安装

使用`git clone`命令下载代码：

```bash
git clone https://github.com/LINC-BIT/BlockTrainer
```

目录结构如下：
- `data`：数据集实现
- `dianzixuebao`：实验代码
- `dnns`：模型实现
- `methods`：算法实现
- `utils`：工具函数

### 2.1 安装依赖

- Ubuntu 18
- Python 3.8+
- CUDA 10.2+

### 2.2 准备环境

首先创建conda虚拟环境：

```bash
conda create -n BlockTrainer python=3.8
conda activate BlockTrainer
```

然后根据[官网](https://pytorch.org/)教程安装`torch`和`torchvision`：
![image](https://user-images.githubusercontent.com/73862727/146364503-5664de5b-24b1-4a85-b342-3d061cd7563f.png)

最后安装本项目所需依赖：

```bash
pip install -r requirements.txt
```

## 3 运行代码

### 3.1 设置

下面使用Vision Transformer (ViT) 作为运行案例，其它的三个模型（Swin / OPT / LLaMA）运行方法相同。

数据集方面，使用ImageNet和SYNSIGNS作为源数据集，使用Caltech256、GTSRB和DomainNet作为目标数据集。

### 3.2 离线准备

依次运行下列命令以准备缩放定律和神经元索引：

```bash
python dianzixuebao/offline_preparing/vit_b_16/img_cls/lora_fine_tune.py
python dianzixuebao/offline_preparing/vit_b_16/img_cls/gen_knowledge_base.py
python dianzixuebao/offline_preparing/vit_b_16/img_cls/gen_neuron_index.py
```

运行后两个命令之前，需将其中的模型路径更改为上一个命令的输出模型路径。

### 3.3 在线训练

运行下列命令以准备缩放定律和神经元索引：

```bash
python dianzixuebao/online_retraining/vit_b_16/img_cls/run.py
```

运行命令之前，需将其中的模型路径更改为离线准备中最后一个命令的输出模型路径。

### 3.4 实验结果

![](https://github.com/LINC-BIT/BlockTrainer/blob/master/docs/eval_result.png)


## 4 架构与实现

![](https://github.com/LINC-BIT/BlockTrainer/blob/master/docs/impl.png)
