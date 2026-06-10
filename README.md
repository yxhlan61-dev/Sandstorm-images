# Sandstorm Weather Image Classification

## 1. 项目简介

这是一个用于 **天气图像三分类** 的课程项目，目标类别为：

- `cloudy`
- `sandstorm`
- `sunny`

目前项目中保留了三条实验路线，方便队友直接比较不同方法的效果：

- `CNN`：基础卷积神经网络分类方法
- `ResNet50`：迁移学习方法
- `Bayes`：传统机器学习方法，基于手工特征

这个仓库不仅保存代码，也保存了实验结果，便于后续：

- 写课程报告
- 做结果对比
- 准备答辩或展示
- 让新加入的队友快速接手

## 2. 仓库结构

```text
Sandstorm Picture/
|- data/
|  |- test/
|  |- train/                 # 本地训练数据，不上传 GitHub
|  `- val/                   # 本地验证数据，不上传 GitHub
|- Recognition_CNN.py
|- Recognition_ResNet50.py
|- Recognition_Bayes.py
|- Predict.py
|- GradCam_updated.py
|- generate_experiment_summaries.py
|- result_CNN/
|- result_resnet/
|- result_bayes/
|- .gitignore
`- README.md
```

## 3. 各文件作用

### `Recognition_CNN.py`

这是最早的 CNN 基线模型训练脚本。

主要功能：

- 读取 `train / val / test` 数据集
- 对图像做基础增强
- 训练一个三分类 CNN
- 输出测试集评估结果
- 生成分类报告、混淆矩阵、学习曲线

这是整个项目最基础的神经网络版本。

### `Recognition_ResNet50.py`

这是基于 `ResNet50` 的迁移学习版本。

主要功能：

- 加载 ImageNet 预训练权重
- 先冻结主干网络，只训练分类头
- 再解冻一部分高层做微调
- 保存测试集评估结果到 `result_resnet/`

这是对 CNN 基线的主要改进版本。

### `Recognition_Bayes.py`

这是传统机器学习对比方案，不依赖深层神经网络。

主要功能：

- 从图像中提取手工特征
- 结合颜色统计和纹理特征
- 使用 `GaussianNB` 完成三分类
- 输出 Bayes 方法的评估结果

它的意义是和神经网络方法做横向对比。

### `Predict.py`

用于加载已经训练好的 CNN 模型并进行预测。

支持两种方式：

- 单张图像预测
- 文件夹批量预测

适合做演示和快速测试。

### `GradCam_updated.py`

用于生成 `Grad-CAM` 热力图，帮助观察 CNN 在预测时关注了图像的哪些区域。

适合做模型可解释性展示。

### `generate_experiment_summaries.py`

用于整理实验结果，自动生成说明材料或结果汇总。

适合在写报告或做 PPT 前统一整理实验输出。

## 4. 结果目录说明

### `result_CNN/`

保存 CNN 相关结果，例如：

- 分类报告
- 混淆矩阵
- 学习曲线
- 评估图
- 示例图
- 结果说明文档

### `result_resnet/`

保存 ResNet50 相关结果，例如：

- 分类报告
- 混淆矩阵
- 学习曲线
- 训练历史
- 结果说明文档

说明：

- 模型文件在本地保留
- 模型文件不上传 GitHub

### `result_bayes/`

保存 Bayes 方法相关结果，例如：

- 分类报告
- 混淆矩阵
- 学习曲线
- 样例预测图
- 评估图
- 结果说明文档

## 5. 数据集组织方式

数据目录采用按划分、按类别分文件夹的结构：

```text
data/
|- train/
|  |- cloudy/
|  |- sandstorm/
|  `- sunny/
|- val/
|  |- cloudy/
|  |- sandstorm/
|  `- sunny/
`- test/
   |- cloudy/
   |- sandstorm/
   `- sunny/
```

说明：

- `train` 和 `val` 只保留在本地，不上传 GitHub
- `test` 可以作为展示或测试样例使用
- 这样做是为了避免仓库体积过大

## 6. 环境依赖

建议使用 Python 3.10 及以上版本。

本项目主要依赖：

- `tensorflow`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pillow`
- `opencv-python`
- `joblib`

可以使用下面的命令安装：

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow opencv-python joblib
```

## 7. 运行方式

### 7.1 训练 CNN

```bash
python Recognition_CNN.py
```

### 7.2 训练 ResNet50

```bash
python Recognition_ResNet50.py
```

### 7.3 运行 Bayes 方法

```bash
python Recognition_Bayes.py
```

### 7.4 使用 CNN 做预测

```bash
python Predict.py
```

### 7.5 生成 Grad-CAM 可视化

```bash
python GradCam_updated.py
```

### 7.6 整理实验说明材料

```bash
python generate_experiment_summaries.py
```

## 8. GitHub 中保留什么

会上传到 GitHub 的内容：

- 代码文件
- 结果图
- 分类报告
- 评估结果
- 说明文档
- README

不会上传到 GitHub 的内容：

- `data/train/`
- `data/val/`
- 模型文件，例如 `.h5`、`.keras`、`.joblib`
- 缓存文件，例如 `.npz`
- 虚拟环境
- Python 缓存

这样做的目的是：

- 方便协作
- 控制仓库大小
- 避免大文件频繁上传

## 9. 队友建议阅读顺序

如果队友第一次接手这个项目，建议按下面顺序阅读：

1. 先看本 `README.md`
2. 再看 `Recognition_CNN.py`，理解基础方法
3. 再看 `Recognition_ResNet50.py`，理解迁移学习改进
4. 再看 `Recognition_Bayes.py`，理解传统方法对比
5. 最后查看 `result_CNN/`、`result_resnet/`、`result_bayes/` 中的结果

这样可以最快建立对整个项目的整体认识。

## 10. 项目当前用途

这个仓库目前适合用于：

- 课程作业协作
- 模型对比实验
- 报告写作
- PPT 展示
- 答辩准备

如果后续需要，我还可以继续把 README 扩展成：

- 更适合课程报告引用的中文版本
- 更适合开发协作的技术说明版本
- 更适合展示的精简版说明
