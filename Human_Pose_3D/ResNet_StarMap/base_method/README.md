本篇主要介绍目前项目中使用的2D姿态估计 和 2D-3D的姿态估计方法；

主要针对 Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach
【 https://arxiv.org/pdf/1704.02447.pdf 】 ，文章进行网络结构的修改及改进；

1.上述论文中描述的网络其实结构比较简单，总的结构就是 stacked hourglass网络 + 深度回归层
2.其中，2d姿态估计使用的是 stacked hourglass网络；
3.该网络的具体描述，在之前的 2D姿态估计中其实已经进行了算法的描述，和网络的具体实现；该
网络的具体的实现参考 stacked_hourglass.py
4.主要对论文中模型进行了复现，包括 2D姿态估计网络以及 2D映射为3D姿态的网络的实现，主要
参考目录下的 hg_3d.py文件；
