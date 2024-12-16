---
layout: post
title: A short review of point cloud detection algorithms - From PointNet to VoxelNeXt (Simplified Chinese)
date: 2023-08-17 11:12:00-0400
description: I have no experience translating it into English and publishing it.
tags: AD
categories: selfstudy-note
related_posts: false
published: true
---

# 摘要

近年来，激光雷达点云目标检测算法在自动驾驶、机器人感知等领域中扮演着重要角色。本综述旨在回顾激光雷达点云目标检测算法的发展历程，重点关注从经典的PointNet算法到最新的VoxelNext算法的演进过程。我将从算法的基本原理、方法优劣势、实验结果等方面综合评述各个算法，并提出当前研究的挑战和未来发展方向。作为激光雷达点云目标检测算法的学习启蒙。

# 引言

## 背景

自动驾驶和机器人技术的快速发展推动了激光雷达点云目标检测算法的研究。激光雷达点云作为一种重要的三维感知数据，具有高精度和丰富的空间信息，因此被广泛应用于目标检测任务中。感知环节，又可以细分为多个模块。具体来讲，从 Detection, Postion, On-board Map 三个方面考虑。本文主要考虑的是其中的 Detection 方面。该环节主要考虑如何实时检测障碍物、目标物等。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/引言.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    3D激光雷达点云目标检测
</div>

- 2016年，PointNet首次提出点云分割的基本思路、骨干网络，以严谨而全面的实验验证算法各个环节的有效性。
- 2017年，VoxelNet以 PointNet 为基础，用 Voxel 描述点云空间，提出VFE（Voxel Feature Encoder，体素特征提取器），作为往后几乎一切点云目标检测算法的首要环节。
- 2018年，SECOND使用改进的稀疏卷积和新的损失函数，对 VoxelNet 进行了一些优化和完善，采用改良的稀疏卷积，效果改善显著。
- 2019年，PointPillar 将VoxelNet中Voxel改为长条形的Pillar，采用SECOND中的稀疏卷积核，几乎不影响性能的情况下极大地提升了性能，是工业界常用的算法之一。
- 2020年，Voxel R-CNN基于voxel特征，设计了3D点云目标检测的two stage网络，提升了些许检测精度。
- 2021年，CenterPoint继承 PointPillar 算法的Pillar化思路，并结合2D的CenterNet算法，引入注意力机制，看到了在3D检测上采取2D检测策略的优势性，进一步提升two stage系列算法性能。
- 2023年，Voxel NeXt回归到点云数据本质，采用纯稀疏的卷积网络，解决了CenterPoint中注意力浪费的部分，并结合Segment Anything开发出一套强鲁棒检测算法驱动的自动标注系统。

以上是近年来一些代表性比较强的点云目标检测算法的发展历程。本文将对其中PointNet，VoxelNet，CenterPoint 以及 VoxelNeXt四个算法进行详细的介绍，并分析算法的创新点，以及一些思考。

## 问题定义

3D目标检测是自动驾驶的一个基础部分，3D目标检测网络通常输入稀疏点云，输出物体三维位置以及类别。硬件设备常常选用激光雷达Lidar。

对于整个点云数据帧$F$，其内部有点集$$P = \{p_1,p_2,...,p_n\}, p_k=\{x_k,y_k,z_k,I_k\} , p_k \in P$$（$$I$$表示点的强度信息）。对$$P$$执行检测算法$$A$$，返回预测目标集$$D=\{d_1,d_2,...,d_m\}, d_i = \{x_i,y_i,z_i,h,w,l,rot,lab\}, d_i\in D$$，其中$$x,y,z$$表示3D框的中心点，$$h,w,l$$表示3D框的高、宽、长，$$rot$$表示在6轴坐标系下的$$yaw$$旋转角，$$lab$$表示3D框的语义信息，即预测目标的类别。

## 评估标准

### Metrics based on Average Precision

检测算法的评估，一般采用 IoU (Inner of Union) 作为评价 BBox 和 Ground-Truth 的重合程度。数学建模为，对于同一个对象$O$，取算法预测框$D$与真值框$G$ (Ground Truth) ，按如下公式计算：

$$IoU = \frac{area(D)\cap area(G)}{area(D)\cup area(G)} $$

定义 $$IoU>0.5$$为真阳性 $$TP$$ (True Positive)，$$IoU\le0.5$$为假阳性$$FP$$（False Positive）；定义真值未检测的为假阴性$FN$(False Negivate)，定义检测出但真值不存在的为真阴性$$TN$$(True Positive)。根据以上几个定义，可以计算出模型整体的准确度$P$(Precision) 与召回率$$R$$(Recall)，并绘制成 $$P-R$$ 曲线。

$$P=\frac{TP}{FP+FN}\ \ \  R=\frac{TP}{TP+FN}$$

计算曲线的下积分面积$$AP$$(Average Precision)。近似地，用以下公式估计$$AP$$的值，表示为对于此次测试的数据，模型整体的平均准确度。对所有测试数据求平均，求的$$mAP$$用于评估整个模型在测试集上的效果。

$$AP = \sum_{k=0}^{k = n - 1}[Recall(k) - Recall(k-1)] \times Precision(k)$$
$$mAP = \frac{1}{n} \sum_{k = 1}^{k = n}AP_k$$

进一步地，更改$$IoU$$的阈值，从 $$0.5$$以$$0.05$$为单位递增至$$0.95$$，计算每个阈值下的$$mAP$$，更严格地评价模型效果，将该指标称为$mAP50-95$。
  对于点云数据，由于其数据特有的稀疏性，通过考虑地平数据点比较稀疏和离散，不好作特征提取，由面上的 2D 中心距离而不是基于联合的亲和力的交集来定义匹配。具体来说，将预测与具有最小中心距离且达到特定阈值的地面真实对象进行匹配。对于给定的匹配阈值，我们通过整合召回率与精度曲线（召回率和精度 > 0.1）来计算$AP$。我们最终对 $$\{0.5,1,2,4\}$$ 米的匹配阈值进行平均，并计算各个类别的平均值。

### Metrics based on True Positive

依据nuScenes提出的标准，定义一组真阳性 (TP) 的指标，用于测量平移、尺度、方向、速度和属性错误。所有$$TP$$指标都是在匹配时使用$2m$中心距的阈值计算的，并且它们都被设计为正标量。

每个类别的匹配和评分都是独立进行的，每个指标都是每个达到 10% 以上的召回水平的累积平均值的平均值。如果特定类别未达到 10\% 的召回率，则该类别的所有 $$TP$$ 错误都将设置为 1。我们定义以下 $$TP$$ 错误：

- 平均平移误差 ATE (Average Translation Error)：二维欧几里得中心距离（以米为单位）；
- 平均尺度误差 ASE (Average Scale Error)：在对齐中心和方向后计算为$$1 - IOU$$ ；
- 平均方向误差 AOE(Average Orientation Error)：预测与地面实况之间的最小偏航角差异（以弧度为单位）。对于所有类别，方向误差均以 360 度进行评估，障碍除外，障碍物仅以 $$180$$度进行评估。忽略锥体的方向错误；
- 平均速度误差 AVE (Average Velocity Error)：绝对速度误差，以 $$m/s$$ 为单位。障碍物和锥体的速度误差被忽略；
- 平均属性误差 AAE (Average Attribute Error)：计算为$$1 - acc$$，其中 $acc$ 是属性分类精度。障碍物和锥体的属性错误被忽略。

以上5个标准与mAP加权求和，即nuScenes定义的NDS (nuScenes Detection Score) 标准，即

$$\text{NDS} = 1/10[5\text{mAP}+ {\sum}_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))]$$

以上所有误差都 >= 0，但是，对于平移误差和速度误差，误差是无界的，并且可以是任何正值。此类 $TP$ 指标是按类定义的，然后我们取类的平均值来计算 $$mATE, mASE, mAOE, mAVE,mAAE$$。

## 任务难点

由于深度学习数据驱动的特性，就算法训练而言，在数据获取上，相比与其他数据，3D点云数据集的制备和处理非常困难。

对比于 2D 图像数据，处理点云数据具有以下几个难点。在后文中，会分别简要提到如何克服以下几个困难。数据点比较稀疏和离散，不好作特征提取，噪声不好区别；具有更多自由度，不好处理（更多的尺寸种类，相对于坐标周有多个维度的旋转）；数据标注困难，每个Ground Truth（真值框）需要八个点位的标注。

## 常见的公开数据集

针对数据获取的困难，往往采用一些常见的开源点云数据集进行训练。

- KITTI 3D：具备1.5w帧数据，8w个标注的目标；
- nuScenes：具备39w帧雷达，140w个标注目标，23类；
- Waymo：具备200w帧，2260w个标注目标，4类+23类；
- Argoverse2：具备1000个场景，每帧图像平均有75个目标物，30个类别；
- Dense: 不同天气下12000个样本和浓雾中的1500个样本，少见的极端天气数据集；
- SUScape：南方科技大学与Intel公司合作，在深圳采集的数据集，具有40+类别，有效平均实例密度为nuScene的10倍。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/nuScene车.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    nuScenes数据集车辆传感器设置
</div>


如果想要自己制备数据集用来训练模型，往往采用以下的方法。

- 自制真实数据集：机器人采集真实数据，通过 Segement Anything，SUSTechPoint 等工具辅助标注；
- 仿真采集数据集：Carla 等仿真环境，数据标注来源于对仿真环境真值的处理。目前有一种主动式的方针采集思路CARLA-ADA，能够制备比较有效的仿真数据集。

# 算法介绍

## PointNet-2016

PointNet 和 PointNet++ 最早被提出的一类3D点云分割模型 (Lidar Segmentation)，作为点云目标检测的先驱和奠基。

### 算法摘要

由斯坦福大学于 2016 年发表论文 "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"，是点云神经网络的鼻祖，它提出了一种网络结构，可以直接从点云中学习特征。

该文章在分类、语义分割两种任务上做出了对比，并给了理论和实验分析。点云的特点其实非常好理解，只要网络抓住以下三个特点，那么它至少就能作为一个能用的 encoder 。

1. 排列不变性：重排一遍所有点的输入顺序，所表示的还是同一个点云数据，网络的输出应该相同。  
2. 点集之间的交互性：点与点之间有未知的关联性。  
3. 变换不变性：对于某些变换，例如仿射变换，应用在点云上时，不应该改变网络对点云的理解。

### 创新点

基于以上提到的点云数据三个特点，PointNet 对问题的处理是：

1. 排列不变性：该文章使用了对称函数（Symmetry Function），它的特点是对输入集合的顺序不敏感。这种函数非常多，如 maxpooling，加法，向量点积等。PointNet 采用的是 maxpooling 方法来聚合点集信息。  
2. 点集之间的交互性：实际上，对称函数的聚合操作就已经得到了全局的信息，此时将点的特征向量与全局的特征向量 concat 起来，就可以让每个点感知到全局的语义信息了，即所谓的交互性。
3. 变换不变性：只需要对输入做一个标准化操作即可。PointNet 使用网络训练出了 D 维空间的变换矩阵。


### 网络结构

PointNet 网络分为两个部分：分类网络 (Classifiction Network) 和分割网络 (Segmentation Network)。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/pointnet-structure.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PointNet算法网络结构
</div>

#### 分类网络

基本思路：分类网络以 $$n$$ 个点作为输入，对输入做特征变换（变换矩阵），将结果输入给 MLP 做回归。将上一步的结果放在特征空间中做特征变换，再输入给 MLP 回归，对得到的输出进行 Maxpool 挑选最大值，作为整个 3D 点云的全局特征。
分类网络设计 Gloabl Feature，这一步称为 Symmetry Function for Unordered Input，对输入做处理。具体来讲，用一个简单的对称函数聚集每个点的信息：

$$f(\{x_1,...,x_2\}) \approx g(h(x_1),...,h(x_n))$$

对此过程数学建模，$$f$$为目标，$g$ 为设计的对称函数。从公式来看，其基本思路是：对各个点$$x_k$$分别做$h$处理，再将所有处理后的点交由函数$g$处理，以实现排列不变性。在实现中，$$h$$为 MLP，$$g$$为 maxpooling。

#### 分割网络

基本思路：分割网络将经过特征空间变换后的点局部特征 (local) 与全局特征 (global) 拼接，输入给 MLP 处理，对每一个点进行分类。

分割网络获取 Point-wise Feature，这一步称为 Local and Global Information Aggregation，对两个不同维度的特征做拼接。

但由于特征空间中的变换矩阵维度远远大于空间中的变换矩阵维度，在softmax训来时，用一个正则化项，将特这个变换矩阵限制为近似的正交矩阵，即输出尺度归一化。这一步称为 Joint Alignment Network：

$$L_{reg}={||I-AA^T||}^2_F$$

其中$$A$$是维度较小网络预测的特征对齐矩阵。消融实验证明该步骤可以有效优化，使模型效果更稳定。

### 模型效果
数据援引论文原文，指标为点的 mIoU(\%)。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/pointnet效果.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PointNet算法效果
</div>

### 总结

PointNet之所以影响力巨大，并不仅仅是因为它是第一篇点云目标检测文章，更重要的是它的网络很简洁（简洁中蕴含了大量的工作来探寻出简洁这条路）却非常的work，这也就使得它能够成为一个工具，一个为点云表征的encoder工具，应用到更广阔的点云处理任务中。

仅用 MLP+max pooling 就击败了众多SOTA，令人惊讶。另外PointNet在众多细节设计也都进行了理论分析和消融实验验证，保证了严谨性，这也为PointNet后面能够大规模被应用提供了支持。

由于 PointNet 模型只使用了 MLP 和 Maxpooling，所获得的特征是全局的，没有捕获局部结构特在，在细节处理和泛用性都不是很好。为使得特征更关注于“局部”，对 3D 点云进行有重叠的多次降采样，分别对每次采样做特征提取，最后进行拼接，其余思路和 PointNet 类似。


