# 基于 CHASE H$\alpha$ 观测的全日面太阳耀斑探测

本仓库是论文 **"Full-Disk Solar Flare Detection Based on Frequency-Guided Attention and Physics-Aware Dimensionality Reduction"** 的官方实现。

该框架将太阳物理诊断（H$\alpha$ 谱线轮廓）与定制化的 YOLO11 架构相结合，实现了高精度、准实时的太阳耀斑探测。

## 🚀 项目架构

本项目基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 框架进行深度定制开发：

* `/ultralytics`: 修改后的 YOLO 核心源码，支持多通道“物理感知”输入及频率引导注意力机制（Frequency-Guided Attention）。
* `gs_batch.py`: 物理感知降维（高斯拟合）脚本。用于将 118 通道的 H$\alpha$ 光谱数据压缩为 3 通道的物理参数图（线心强度、多普勒速度、半全宽）。

## 📊 数据集说明

由于原始 CHASE 观测数据量巨大（全量数据超过 1 TB），我们采取分级开源策略：

1. **公开下载 (Google Drive)**:
* **预处理数据**: 经过高斯拟合压缩后的 3 通道物理参数图。
* **标注数据**: 经过专家修正的 YOLO 格式耀斑检测标签。
* *下载链接: [YOLO 格式数据](https://drive.google.com/file/d/1Pc3fXlNYBFqidHVZNHbaxJkN7iEiN27T/view?usp=drive_link)*


2. **原始数据与全量集**:
* 我们在网盘中提供了一个**样例原始数据**（单幅全日面 FITS 文件），用于测试 `gs_batch.py` 压缩代码。
* **118通道全量数据**: 由于体积超过 1TB，上传下载成本极高。如需全量原始数据进行研究，请通过邮件联系我们。



## 🛠 安装与使用

### 1. 环境配置

```bash
pip install -e .

```

### 2. 数据压缩（物理降维）

将原始 CHASE FITS 文件转换为物理参数图：

```bash
python gs_batch.py

```

### 3. 训练与推理

项目核心已封装在 ultralytics 目录下，使用方式与标准 YOLO 一致。

