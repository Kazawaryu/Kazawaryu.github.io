---
layout: post
title: A Review of 3D Point Cloud Detection Algorithms - from PointNet to VoxelNeXt
date: 2023-08-17 11:12:00-0400
description: Please note that it is written in Simplified Chinese
tags: AD survey
categories: note
thumbnail: assets/img/blog/pcd-survey/引言.png
related_posts: false
published: true
---

> This article introduces some common 3D point cloud detection algorithms (as of August 2023). The content comes from personal understanding and may have some problems.

# 1. 摘要

近年来，激光雷达点云目标检测算法在自动驾驶、机器人感知等领域中扮演着重要角色。本综述旨在回顾激光雷达点云目标检测算法的发展历程，重点关注从经典的PointNet算法到最新的VoxelNext算法的演进过程。我将从算法的基本原理、方法优劣势、实验结果等方面综合评述各个算法，并提出当前研究的挑战和未来发展方向。作为激光雷达点云目标检测算法的学习启蒙。

# 2. 引言

## 2.1 背景

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

## 2.2 问题定义

3D目标检测是自动驾驶的一个基础部分，3D目标检测网络通常输入稀疏点云，输出物体三维位置以及类别。硬件设备常常选用激光雷达Lidar。

对于整个点云数据帧$F$，其内部有点集$$P = \{p_1,p_2,...,p_n\}, p_k=\{x_k,y_k,z_k,I_k\} , p_k \in P$$（$$I$$表示点的强度信息）。对$$P$$执行检测算法$$A$$，返回预测目标集$$D=\{d_1,d_2,...,d_m\}, d_i = \{x_i,y_i,z_i,h,w,l,rot,lab\}, d_i\in D$$，其中$$x,y,z$$表示3D框的中心点，$$h,w,l$$表示3D框的高、宽、长，$$rot$$表示在6轴坐标系下的$$yaw$$旋转角，$$lab$$表示3D框的语义信息，即预测目标的类别。

## 2.3 评估标准

### 2.3.1 Metrics based on Average Precision

检测算法的评估，一般采用 IoU (Inner of Union) 作为评价 BBox 和 Ground-Truth 的重合程度。数学建模为，对于同一个对象$$O$$，取算法预测框$D$与真值框$$G$$ (Ground Truth) ，按如下公式计算：

$$IoU = \frac{area(D)\cap area(G)}{area(D)\cup area(G)} $$

定义 $$IoU>0.5$$为真阳性 $$TP$$ (True Positive)，$$IoU\le0.5$$为假阳性$$FP$$（False Positive）；定义真值未检测的为假阴性$FN$(False Negivate)，定义检测出但真值不存在的为真阴性$$TN$$(True Positive)。根据以上几个定义，可以计算出模型整体的准确度$P$(Precision) 与召回率$$R$$(Recall)，并绘制成 $$P-R$$ 曲线。

$$P=\frac{TP}{FP+FN}\ \ \  R=\frac{TP}{TP+FN}$$

计算曲线的下积分面积$$AP$$(Average Precision)。近似地，用以下公式估计$$AP$$的值，表示为对于此次测试的数据，模型整体的平均准确度。对所有测试数据求平均，求的$$mAP$$用于评估整个模型在测试集上的效果。

$$AP = \sum_{k=0}^{k = n - 1}[Recall(k) - Recall(k-1)] \times Precision(k)$$
$$mAP = \frac{1}{n} \sum_{k = 1}^{k = n}AP_k$$

进一步地，更改$$IoU$$的阈值，从 $$0.5$$以$$0.05$$为单位递增至$$0.95$$，计算每个阈值下的$$mAP$$，更严格地评价模型效果，将该指标称为$mAP50-95$。
  对于点云数据，由于其数据特有的稀疏性，通过考虑地平数据点比较稀疏和离散，不好作特征提取，由面上的 2D 中心距离而不是基于联合的亲和力的交集来定义匹配。具体来说，将预测与具有最小中心距离且达到特定阈值的地面真实对象进行匹配。对于给定的匹配阈值，我们通过整合召回率与精度曲线（召回率和精度 > 0.1）来计算$AP$。我们最终对 $$\{0.5,1,2,4\}$$ 米的匹配阈值进行平均，并计算各个类别的平均值。

### 2.3.2 Metrics based on True Positive

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

## 2.4 任务难点

由于深度学习数据驱动的特性，就算法训练而言，在数据获取上，相比与其他数据，3D点云数据集的制备和处理非常困难。

对比于 2D 图像数据，处理点云数据具有以下几个难点。在后文中，会分别简要提到如何克服以下几个困难。数据点比较稀疏和离散，不好作特征提取，噪声不好区别；具有更多自由度，不好处理（更多的尺寸种类，相对于坐标周有多个维度的旋转）；数据标注困难，每个Ground Truth（真值框）需要八个点位的标注。

## 2.5 常见的公开数据集

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

# 3. 算法介绍

## 3.1 PointNet-2016

PointNet 和 PointNet++ 最早被提出的一类3D点云分割模型 (Lidar Segmentation)，作为点云目标检测的先驱和奠基。

### 3.1.1 算法摘要

由斯坦福大学于 2016 年发表论文 "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"，是点云神经网络的鼻祖，它提出了一种网络结构，可以直接从点云中学习特征。

该文章在分类、语义分割两种任务上做出了对比，并给了理论和实验分析。点云的特点其实非常好理解，只要网络抓住以下三个特点，那么它至少就能作为一个能用的 encoder 。

1. 排列不变性：重排一遍所有点的输入顺序，所表示的还是同一个点云数据，网络的输出应该相同。  
2. 点集之间的交互性：点与点之间有未知的关联性。  
3. 变换不变性：对于某些变换，例如仿射变换，应用在点云上时，不应该改变网络对点云的理解。

### 3.1.2 创新点

基于以上提到的点云数据三个特点，PointNet 对问题的处理是：

1. 排列不变性：该文章使用了对称函数（Symmetry Function），它的特点是对输入集合的顺序不敏感。这种函数非常多，如 maxpooling，加法，向量点积等。PointNet 采用的是 maxpooling 方法来聚合点集信息。  
2. 点集之间的交互性：实际上，对称函数的聚合操作就已经得到了全局的信息，此时将点的特征向量与全局的特征向量 concat 起来，就可以让每个点感知到全局的语义信息了，即所谓的交互性。
3. 变换不变性：只需要对输入做一个标准化操作即可。PointNet 使用网络训练出了 D 维空间的变换矩阵。


### 3.1.3 网络结构

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

### 3.1.4 模型效果
数据援引论文原文，指标为点的 mIoU(\%)。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/pointnet效果.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    PointNet算法效果
</div>

### 3.1.5 算法总结

PointNet之所以影响力巨大，并不仅仅是因为它是第一篇点云目标检测文章，更重要的是它的网络很简洁（简洁中蕴含了大量的工作来探寻出简洁这条路）却非常的work，这也就使得它能够成为一个工具，一个为点云表征的encoder工具，应用到更广阔的点云处理任务中。

仅用 MLP+max pooling 就击败了众多SOTA，令人惊讶。另外PointNet在众多细节设计也都进行了理论分析和消融实验验证，保证了严谨性，这也为PointNet后面能够大规模被应用提供了支持。

由于 PointNet 模型只使用了 MLP 和 Maxpooling，所获得的特征是全局的，没有捕获局部结构特在，在细节处理和泛用性都不是很好。为使得特征更关注于“局部”，对 3D 点云进行有重叠的多次降采样，分别对每次采样做特征提取，最后进行拼接，其余思路和 PointNet 类似。

## 3.2 VoxelNet-2017

### 3.2.1 算法摘要

由 Apple 公司于 2017 年发表论文 "VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection"，是 3D 点云目标检测真正利用好体素化 (Voxel) 的第一篇文章。

在此之前，3D 点云目标检测的主流方法是：(1).数据降维：3D 数据投影成 2D 图像，用传统目标检测方法计算 (2). 整个点云 Voxel 分割后手工设计特征。这些方法对点云数据的利用不足。

VoxelNet 属于单阶段，端到端的点云检测。大致流程为：按空间位置划分 Voxel，然后将 Voxel 内部的点云进行 VFE (Voxl feature encoding) 编码，再接入RPN来生成检测结果，在KITTI数据集上表现出色。

### 3.2.2 创新点

对于 VoxelNet 模型，其突破点和创新点是用 Voxel 描述空间，使用 PointNet 网络对 Voxel 进行特征提取，用该特征代表每个 Voxel ，并放回 3D 空间中，使点云数据有序化。完成三维的空间的特征描述后，对 Voxel 做三维卷积，在得到的特征图上进行目标检测。其实现方式可以说成是，对每个voxel使用PointNet得到voxel的feature。

### 3.2.3 网络结构

VoxelNet 网络分为三个层级：特征学习网络 (Feature Learning Network)、卷积中间层 (Convolutional MIddle Layers)、区域候选网络 (Region Proposal Network)。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/voxelnet结构.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    VoxelNet算法网络结构
</div>

#### 特征学习网络

通过以下五个步骤，提取和学习 3D 点云数据特征。Voxel 化点云、聚合和随机采样很好地解决了“数据点比较稀疏和离散，不好作特征提取”的问题。

1. Voxel 划分：在雷达坐标系中，对于空间 $$D\{X,Y,Z\}$$（针对要检测的物体，会切割出不同的长方体），按$$V\{v_x,v_y,v_z\}$$为单位，划分为小的Voxel，有$$Z'=Z/v_z, Y'=Y/v_y, X'=X/v_x$$。
2. 聚合：由于雷达数据的离散性，点云在三维空间内的分布不均匀，也有相当可能出现空检，将相邻的Voxel进行聚合，尽量减少这种对网络训练不利的情况出现。
3. 随机采样：对于聚合后的Voxel，随机在非空的Voxel内采样$$T$$个点。这一步后，将点云数据表示为$$\{N,T,C\}$$，$$N$$为非空Voxel个数，$$T$$为每个Voxel内的随机采样点个数 ，$$C$$为点的特征。对于不足$$T$$个点的 Voxel，采用 “高斯补0” 算法。
4. VFE（Voxel Feature Enocding）堆叠：
    1. 对于Voxel，首先数据增强其中的每一个点，计算平均值，再计算每个点的偏移量，与原始数据拼接作为输入（Point-wise Input）；
    2. 采用PointNet模型的方法，将Voxel中的点通过全连接层（FCN）转化到特征空间 (Point-wise Feature)，在新的特征中挑选出特征值最大的点 (Element-wise Maxpool)，作为每个Voxel的表面形状特征 (Locally Aggregated Feature)； (c). 获取每个Voxel的特征后，最后再拼接 (Point-wise Concatenate)成为更高维的特征 (Point-wise concatenated Feature)。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/VFE.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    VFE：体素特征编码器的编码过程
</div>

特征提取后稀疏特征的表示：上一步中，都是对非空的 voxel 进行处理，这些 voxel 仅仅对应 3D 空间中很小的一部分空间。这里需要将得到的 N 个非空的 voxel 特征重新映射回来源的3D空间中，表示成一个稀疏的 4D张量，$$（C，Z'，Y'，X'）-> (128, 10, 400, 352) $$。这种稀疏的表示方法极大的减少了内存消耗和反向传播中的计算消耗。同时也是 VoxelNet 为了效率而实现的重要步骤。

#### 卷积中间层

简单来讲，用以下三个三维卷积核对 Voxel 化的点云进行卷积，每个卷积后都接 BN 层 (Batch Normalization) 和 ReLU 层，增强传递防止梯度消失，归一化加速网络收敛。通过卷积中间层，能够提升感受视野

$$Conv3D_1(128, 64, 3, (2,1,1), (1,1,1))$$
$$Conv3D_2(64, 64, 3, (1,1,1), (0,1,1))，$$
$$Conv3D_3(64, 64, 3, (2,1,1), (1,1,1))$$

通过中间层运算之后，4D的tensor尺寸为（拿car detection为例，62*2*400*352），然后会进行reshape操作，将特征图变成 128*400*352便于后续使用RPN。

#### 区域候选层

RPN层的概念在FasterRCNN中就被提出来，主要是用于根据特征图中学习到的特征和结合anchor来生成对应的预测结果。VoxelNet的预测头，类似于SSD和YOLO那一类的目标检测算法种的头预测。在 FrCNN 中 RPN 在每个像素和像素的中心点位置根据 anchor 的设置，预测了一个 anchor 属于类别，以及针对该 anchor 的粗回归调整。在VoxelNet 中也不例外。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/RPN.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    RPN：区域候选网络的特征堆叠过程
</div>

VoxelNet 的 RPN 结构在经过前面的 Convolutional middle layers 和 tensor 重组得到的特征图后，对这个特征图分别的进行了多次下采样，然后再将不同下采样的特征进行反卷积操作，变成相同大小的特征图。再拼接这些来自不同尺度的特征图，用于最后的检测。给人的感觉类似图像目标检测的 NECK 模块中 PAN，只不过这里只有一张特征图。将不同尺度的信息融合在了一起。这里每一层卷积都是二维的卷积操作，每个卷积后面都接一个BN和RELU层。输出的结果是一个分类预测和 anchor 回归预测的结果。

### 3.2.4 锚点选择与损失函数

锚点的选择采用（拿car detection举例）选取 $$l^a=3.9,w^a=1.6,h^a=1.59,z^a=-1.0,\theta=[0,90]$$。定义$$\{a_i^{pos}\}_{i=1...N_{pos}}$$为正样本锚点，$$\{a_i^{neg}\}_{i=1...N_{neg}}$$为负样本锚点。检测框被参数化成 $$u=(x,y,z,l,w,h,\theta)$$。

Anchol与真实值之间的匹配方案为（拿car detection举例），看其在BEV下面的IOU值，当IOU最大时为正样本，或者$$IOU > 0.6$$ 为正样本，$IOU<0.45$为负样本，介于$$0.45$$与$$0.6$$之间的丢弃掉。最终的残差形式为下面的函数，其中前两项为分类的损失，后一项为框的拟合程度损失。

$$L=\alpha\frac{1}{N_{pos}}\sum_{i}L_{cls}(P_i^{pos}, 1)+\beta\frac{1}{N_{neg}}\sum_jL_{cls}(p_j^neg, 0)+\frac{1}{N_{pos}}\sum_iL_{reg}(\textbf{u}_i,\textbf{u}_i^*)$$

训练模型前，论文对数据进行了一定程度的数据增强。对真实的3D标注框，进行旋转，平移，同时去除的碰撞的情况。以及对3D标注框，整体点云分别进行$$[0.95~1.05]$$之间不同的缩放。

### 3.2.5 模型效果

数据援引论文原文，训练和测试均在KITTI上进行，指标为ACC。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/voxelnet效果.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    VoxelNet模型在KITTI数据集上的效果
</div>

### 3.2.6 算法总结

VoxelNet 模型属于比较早期的将点云转为 Voxel 作处理的模型（2017），突破性地真正发挥 Voxel 分割点云的作用。此后，在这个思路上，对点云目标检测的效果进行了许多改进。更关键的，论文提出了一种基于体素的编码器VFE（Voxel Feature Encoder），日后几乎所有的3D目标检测模型都采用VFE作为处理点云的第一个步骤，这足以说明VoxelNet所产生的影响之深。

## 3.3 CenterPoint-2021

### 3.3.1 论文摘要

由 Utexas 于 2021 年发表论文 “Center-based 3D Object Detection and Tracking”，采用前人改进的骨干网络，将2D CenterPoint 算法推广到 3D CenterPoint 算法。

其基本思路是：1. 点云数据通过3D骨骼网络，以Voxel特征描述，提取BEV下的特征图； 2. 基于2D的RCNN检测头寻找目标中心点，以及边框的3D尺寸，3D朝向，速度（stage one），依据中心点以中心特征回归进一步优化预测值（stage two）。其本质是一个two stage, anchor free的算法。

在nuScenes上表现出SOTA水准（NDS 65.5，AMOTA 63.8 ），同时在waymo数据集上远好于其他纯lidar方案。作者认为，使用中心点表示，能够大幅减少3D检测的难度，在应对不同朝向上有很好的表现。

骨骼网络采用 Voxel-Net 或 PointPillar，将空间分割为长条形的Voxel（Pillar），输入给Two-Stage模型预测。

### 3.3.2 创新点

CenterPoint 提出使用点来代表三维空间中的物体，使用一个关键点检测器来物体中心，以及回归属性（3D size，3D朝向，速度）。在第二阶段，借助点特征来进一步优化这些估计。

利用点的表示形式，避免了主干网络去学习旋转不变性。同时引入了CneterNet算法中的注意力机制，提升算法表现。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/centerpoint注意力.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    CenterPoint利用了CenterNet算法的注意力机制
</div>

简单来讲，CenterPoint 用 Voxel 描述空间，用 Point 描述物体；结合 2D 目标检测经验，增加单位数据价值；减少 Voxel 计算量，进一步解决“数据处理点多，平均一次扫描获取几十万个数据点”的问题。

### 3.3.3 网络结构

CenterPoint 网络分为三个层级：3D 骨骼网络 (3D Backbone Network)、区域候选网络 (Region Proposal Network)、区域卷积神经网络 (Regions with CNN features)。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/centerpoint结构.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    CenterPoint算法网络结构
</div>

#### 3D 骨干网络

采用骨干网络 (VoxelNet 或 PointPillar)。思路比较类似，将 3D 点云划分为 Voxel，但在$Z$方向上不做划分，感官上类似于“长条形”，用这类 Voxel（Pillar） 描述点云空间，提取点云特征。

提取到的特征统称为 map-view feature，$$M\in\mathbb{R}^{W*L*F}$$ (一个3D的Tensor，类比图像的$$W*H*C$$，这样就就可以采用图像的方法)。

#### 多种检测头 - Stage One

3D 骨骼网络输出了点云特征，在 Stage One 通过3种检测头来得到 (1). center heatmap head（检测中心点），(2).regression head （检测中心点偏移+边框）(3). velocity head （检测目标速度）。

1. Center heatmap head: 与 CenterNet 思路类i，在 backbone 特征后连接一个 heatmap header 用于预测目标中心点。对于$$k$$个label，输出$k$个channel。选取 top100 个热力值的 peak 点作为正负样本的候选。考虑到点云数据稀疏，将gt的框投影到map-view feature中时，以gauss focal loss的方式，扩大对每个gt的高斯渐变范围，其半径为$$\delta =max(f(wl),\tau)$$，增多监督学习中的正样本数量。
2. Regression head: 回归中心点的偏差（中心点归属的voxel存在取整后的误差），以及3DBBox信息，训练阶段仅正样本计算损失，推理时选取 heatmap 峰值，在 regression head 中获取 3D-BBox 的位置。
3. Velocity head: 预测二维的速度，用于tracking，这个需要输入当前时刻与上一时刻的map-view feature，只做检测可以不采用这一环节。预测的方式采用贪心匹配（匈牙利算法，类似SORT），如果目标连续三帧无匹配，则删除映射。

#### 预测 - Stage Two

将第一阶段检测到的100个框，每一个框投影回map-view feature，拿到4条边中心点+1个中心点，对应的feature，堆叠起来。
为解决坐标轴对齐问题，采用双线性插值来获取以上点的特征。
最后，整合以上5个特征，输入给MLP计算，预测置信度和回归信息，分类信息在 Stage one 已经解决，得到带有方向的预测bbox。


置信度损失计算: 检测框的目标置信度采用下面方式计算，基本上>0.5则目标置信度为1，<0.25则置信度为0，然后结合交叉熵来计算损失。

$$L_{score}=-I_t\log(\hat{I}_t)-(1-I_t)\log(1-\hat{I}_t)$$
$$I=min(1,max(0,2 \times IoU_t-0.5))$$

在推理阶段，置信度的计算方式如下，其中$$\hat{Y}_t=max_{0\le k \le K}\hat{Y}_{p, k}$$和$$\hat{I}_t$$分别表示 stage one 和 stage two 对目标$$t$$的置信度。

$$\hat{Q}_t=\sqrt{\hat{Y}_t\times\hat{I}_t} $$

边框回归：采用NMS，和2D目标检测的思路基本一致，属于比较经典的做法。

### 3.3.4 模型效果

数据援引论文原文，训练和测试均在Waymo上进行，指标为mAP和mAPH。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/centerpoint表现.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    CenterPoint模型对比之前的所有主流模型
</div>

### 3.3.5 算法总结

单阶段的模型效果很好。在2021年达到SOTA水平（Waymo和nuScenes），性能优于3DSSD，PointPillar。

但运行速度较慢，在Titan RTX上实验，Waymo 11FPS， nuScenes 16FPS， 勉强达到实时计算要求。部署在真车上仍需要改进（数据援引原论文）；在最新的论文中显示，平均的latency为96ms (CVPR2023)。

## 3.4 VoxelNeXt-2023

由 CUHK 于 2023 年3 月发表论文 "VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking"，真正直接采用点云数据作预测。

### 3.4.1 创新点

目前的三维检测网络，通常使用稀疏卷积来提取特征（出于效率的考虑），借鉴二维的物体检测网络，锚点，或者中心点（CenterPoint网络）被普遍采用。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/voxelnext注意力.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    CenterPoint模型的注意力模块的计算存在不少性能浪费
</div>

锚点或者中心点，都是为图像设计的，没有考虑到点云的不规则以及稀疏性，为了应用这些替代的表示方法，主流的检测器将三维稀疏特征转换成二维的稠密特征，然后使用RPN，以及使用稠密的检测头。尽管有效，但这样低效以及复杂。从 CenterPoint 的热力图可以看出，很多部分的 BEVfeature 计算都是浪费的。后续还需要进行 NMS来去除重复的检测。

为解决 CenterPoint 热力图和 NMS 重复计算问题，并进一步发掘激光雷达数据稀疏性的特点，提出 VoxelNeXt 模型，采用纯稀疏的 Voxel 网络，直接基于3D点云预测，而不是通过2D数据升维预测。减少了非常多的计算量，让实时的雷达目标检测成为可能。

### 3.4.2 网络结构

在没有任何其他复杂设计的情况下，通过额外的下采样层可以简单地解决感受野瓶颈不足的问题。点云或体素分布不规则，通常分散在3D对象的表面，而不是中心或内部，因此直接基于体素而不是手工制作的锚或中心来预测3D Bbox。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/voxelnext结构.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    对比以往常规的3D目标检测，VoxelNeXt的网络结构非常简单
</div>

VoxelNeXt 网络包含了4个实现细节：1) 骨干网络，2) 将3D稀疏体素压缩成 2D 稀疏体素，3）Sparse max pooling / NMS，4）用3x3 sparse conv或FC来预测物体。

#### 骨干网络

一般情况下，简单稀疏的 CNN 骨干网络有4个阶段，其特征步长为 $$\{1,2,4,8\}$$，将其输出的稀疏特征命名为$$\{F_1,F_2,F_3,F_4\}$$。目前的特征无法描述和预测，尤其是大型的对象（占多个Voxel）。额外多两次下采样，得到步长为$$\{16,32\}$$的特征$$\{F_5,F_6\}$$。

将最后三个步骤$$\{F_4,F_5,F_6\}$$的稀疏特征进行拼接，只需要作简单的稀疏串联，而不需要其他复杂的参数化层，使其空间对其到$$F_4$$特征空间。其中，对于阶段$i$，$F_i$是一组单独的特征$$f_p$$，$$p\in P_i$$是 3D 空间的点，坐标为$$\{x_p,y_p,z_p\}$$。

$$F_c=F_4\cup(F_5\cup F_6)$$
$$P_6'=\{(x_p\times2^2,y_p\times2^2,z_p\times2^2)  |  p\in P_6 \}$$
$$P_5'=\{(x_p\times2^1,y_p\times2^1,z_p\times2^1) | p\in P_6 \}$$
$$P_c=P_4\cup(P_5' \cup P_6')$$

在两次额外下采样下做稀疏特征串联后，有效感受范围扩大，预测更准确，且不需要太多的计算。

#### 体素压缩

3D 体素映射到 2D：这一步骤中，将稀疏特征转换为密集特征，压缩$$z$$方向，将 3D 体素特征压缩为密集的 2D 特征图。VoxelNet 中发现，2D 的稀疏特征对预测有效，不单单只是抑制模型收敛。在 VoxelNeXt 中，对高度的压缩只是以体素为对单位，映射在统一平面上，对于同一区域的特征累加。数学建模以上过程，如下所示：

$$\bar{P}_c=\{(x_p,y_p)\|p\in P_c\}$$
$$\bar{F}_c=\{\sum_{p\in S_{\bar{p}}}{f_p}  \| \bar{p}\in\bar{P}_c\}$$

其中$$S_{\bar{p}}=\{p\|x_p=x_{\bar{p}},y=y_{\bar{p}, p\in P_c} \}$$，包含映射在 2D 平面上的体素。

体素裁减：由于网络完全基于体素本身，而 3D 点云中含有大量冗余的背景点，对预测有很大的不利，因此需要对映射后的 2D 体素做裁剪。沿着下采样层逐渐修剪不相关的体素，根据SPS Conv，抑制了具有小特征量值的体素的膨胀。将抑制比设为0.5，仅对特征幅度$$\|f_p\|$$（在通道维度上平均）排在前一半的体素进行扩张。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/voxelnext下采样.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    类似高维合并后的下采样式体素裁减
</div>

体素选择：论文没有采用常用的 NMS 方法，不依赖于密集的特征图，而是基于3D CNN骨干网络的输出进行预测。在训练过程中，我们将离每个注释边界框中心最近的体素指定为正样本。我们使用焦点损失进行监督。
在推理时，由于特征足够稀疏，可以直接用简单 max pooling 选择具有空间局部最大特征的体素，节省检测头的计算。

回归预测：
回归方法与 CenterPoint 类似，简单地使用核大小为3的全连通层或 3×3 子流形稀疏卷积层进行预测，而不需要其他复杂的设计。论文发现，3×3 稀疏卷积比全连接层产生更好的结果，但目前缺少数学上的理论支撑。
同样的，执行 3D Tracking 任务时的思路和 CenterPoint 类似，使用体素关联来包含更多与查询体素位置匹配的轨迹，不多赘述。

### 3.4.3 模型效果

对比CenterPoint，训练和测试在Waymo数据集上，指标为IoU。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/voxelnext对比centerpoint.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    VoxelNeXt解决了CenterPoint模型中注意力浪费的部分，性能得到提升
</div>

对比以往常规思路的3D点云目标检测算法，训练和测试在Waymo数据集上。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/blog/pcd-survey/voxelnext表现.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    VoxelNeXt与其他模型的效果对比
</div>

### 3.4.4 论文总结

论文思想来源于 CenterPoint，从 3D 的角度考虑点云特征，没有手动构建而是直接送入网络学习，同时很好的发挥了点云数据的稀疏性，减少计算量的同时优化表现，在多个数据集上都做到SOTA。

实际表现，可能在训练上有一定难度。由于模型本身由于没有忽略$z$方向上的信息，对高度敏感，易于在不同高度上检测出物体。

但模型的训练存在比较明显的问题，论文中采用知识蒸馏的方案，同时训练了比较长的时间，得到了比较好的预期结果。实际在本地训练时，耗时巨大且提升不明显，是需要一些高级的模型训练方案才可以得到预期效果的检测模型。

# 4. 实验复现

## 4.1 实验设计

准备两组数据集$$D_{sim}$$和$$D_{real}$$，分别在仿真环境Carla和真实环境（设备Velodyne—VLP16，园区内）采集。分别训练VoxelNet、PointPillar、CenterPoint以及VoxelNeXt模型(epochs=160, batch size=18，split=0.2)。观察训练时间以及表现效果，数据集的详细信息如下：

| **Dataset** | **Size** | **Frame** | **Instances** | **class**                             | **detail** |
|-------------|--------|---------|----------|--------------------------------------|----------|
| $$D_{sim}$$   | 20.03G | 3971    | 75294    | Car, Truck, Van, Pedestrian, Cyclist | city, highway |
| $$D_{real}$$  | 16.77G | 4068    | 66329    | Vehicle, Pedestrian                  | Campus     |

## 4.2 实验设备

- CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
- GPU: NVIDIA TITAN V x 6
- OS: Ubuntu 22.04.2 LTS
- MEM: 453 G

## 4.3 实验结果

以下为仿真数据集试验结果。

| **样本类**    | **VoxelNet** | **PointPillar** | **CenterPoint** | **VoxelNeXt** |
|------------|--------------|-----------------|-----------------|---------------|
| Car        | 44.68        | 60.28           | **64.09**    | 24.09         |
| Truck      | 52.39        | **71.27** | 68.32           | 38.23         |
| Van        | 52.46        | 54.11           | **71.09**  | 30.48         |
| Pedestrian | 30.22        | 40.87           | **48.05**  | 9.71          |
| Cyclist    | 29.14        | **56.30**  | 52.26           | 12.33         |
| 训练时间       | 3h33min      | 4h17min         | 5h39min         | 6h13min       |


以下为真实采集数据集下实验结果。

| **class**    | **VoxelNet** | **PointPillar** | **CenterPoint** | **VoxelNeXt** |
|------------|--------------|-----------------|-----------------|---------------|
| Vehicle    | 22.03        | 26.75           | **28.09**   | 15.92         |
| Pedestrian | 9.51         | 11.20           | **14.69**  | 4.32          |
| time cost       | 3h04min      | 3h45min         | 4h12min         | 6h27min       |

## 4.4 结果分析

1. 就模型训练时间而言，随算法复杂度提高，训练时间逐渐增加。
2. 就模型表现（mAP）而言，在仿真数据集上，CenterPoint算法与PointPillar不相上下。在真实数据集上，CenterPoint算法表现效果最好。
3. 由于真实数据采集所用的设备Velodyne-VLP16是16线激光雷达，对于目标物体的描述很不清晰，导致训练结果很差，与仿真数据（128线模拟激光雷达）所训练的模型表现相差甚远。


# 5. 总结

从PointNet到VoxelNeXt算法，激光雷达3D目标检测算法的发展越见成熟。从提出骨干网络，到VFE和RPN的首个处理过程，到转为two stage算法，到引入注意力机制，激光雷达3D目标检测算法越来越复杂，效果也逐渐提升。然而，离数据本身的特征看似却越来越远。VoxelNeXt的提出无疑是给研究者们一个很好的启示——利用好数据本身的特质，即使是简单的网络结构依然能有非常好的效果，看起来是揭示了激光雷达3D点云目标检测的又一个发展新趋势。

本篇文章从最初的激光雷达3D点云目标检测的诞生讲起，对其发展历程中的几个重大的里程碑式的算法进行了详细的介绍与分析，在本地部署和测试算法性能，最后提出了些许个人看法。希望能够对激光雷达3D点云目标检测算法的初学者带来一些帮助。





