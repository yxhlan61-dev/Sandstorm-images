# 沙尘暴图像识别项目

本仓库用于保存沙尘暴天气图像识别实验的测试样例、训练脚本、预测脚本、模型文件和可视化结果。项目面向三类天气图像分类任务：`cloudy`、`sandstorm` 和 `sunny`。

## 仓库内容

- `data/test/`：仓库中保留的测试集样例，按 `cloudy`、`sandstorm`、`sunny` 三个类别组织。
- `Recognition.py`：使用 TensorFlow/Keras 训练卷积神经网络，并保存优化后的模型。
- `Predict.py`：加载训练好的模型，对单张图片或文件夹中的图片进行预测。
- `GradCam_updated.py`：生成 Grad-CAM 热力图，用于观察模型关注的图像区域。
- `optimized_sandstorm_model.h5`：已经训练好的沙尘暴图像识别模型。
- `result/`：实验结果图、网络结构图和相关可视化材料。

## 数据说明

```text
data/
  test/
    cloudy/
    sandstorm/
    sunny/
```

为控制仓库体积，GitHub 仓库中只上传 `data/test`。本地训练时仍可使用 `data/train` 和 `data/val`，但这两个目录默认不纳入版本管理。

## 环境依赖

建议使用 Python 虚拟环境安装依赖。本项目主要依赖：

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn seaborn
```

## 训练模型

运行以下命令开始训练：

```bash
python Recognition.py
```

训练脚本默认读取本地的 `data/train`、`data/val` 和 `data/test` 数据，完成模型训练、评估和可视化，并将模型保存为 `optimized_sandstorm_model.h5`。

## 图像预测

运行预测脚本：

```bash
python Predict.py
```

脚本支持两种方式：

- 输入单张图片路径，输出预测类别和置信度。
- 输入文件夹路径，批量预测文件夹中的图片。

## Grad-CAM 可视化

运行：

```bash
python GradCam_updated.py
```

该脚本会加载模型并生成热力图，帮助分析模型判断某张图片类别时重点关注的区域。使用前可在脚本末尾修改 `img_path`，指定需要解释的测试图片。

## 版本管理说明

本仓库会忽略本地虚拟环境、IDE 配置、Python 缓存、训练集、验证集和临时输出文件。模型文件和视频等较大的二进制文件通过 Git LFS 管理。

## 项目用途

该项目可用于天气图像分类、沙尘暴识别实验、卷积神经网络训练流程演示，以及 Grad-CAM 模型可解释性分析。
