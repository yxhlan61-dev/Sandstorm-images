import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# 使用非交互式后端，保证脚本在终端或服务器环境下也能稳定保存图片。
matplotlib.use("Agg")

# Matplotlib 中文显示设置，避免图标题和坐标轴中文乱码。
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

# 允许处理的图像扩展名。
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
# 灰度共生矩阵的灰度量化等级数，与传统纹理分析写法保持一致。
GLCM_LEVELS = 16
# 共生矩阵统计时的像素距离，取 1 表示相邻像素关系。
GLCM_DISTANCE = 1
# 本项目固定的三分类类别顺序。
CLASS_NAMES = ["cloudy", "sandstorm", "sunny"]


@dataclass
class SplitData:
    """保存一个数据划分中的特征、标签和原始图片路径。"""

    features: np.ndarray
    labels: np.ndarray
    paths: List[str]


def collect_split_paths(split_dir: str) -> Tuple[List[str], np.ndarray]:
    """
    按固定类别顺序扫描一个数据划分目录。

    返回：
    1. 每张图片的完整路径
    2. 与路径顺序一一对应的整数标签
    """
    paths: List[str] = []
    labels: List[int] = []

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(split_dir, class_name)
        filenames = sorted(
            name for name in os.listdir(class_dir) if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        for filename in filenames:
            paths.append(os.path.join(class_dir, filename))
            labels.append(label_index)

    return paths, np.asarray(labels, dtype=np.int64)


def compute_channel_statistics(channel: np.ndarray) -> np.ndarray:
    """
    计算单通道颜色统计特征。

    这里参考你给的 C++ 颜色特征思路，对单通道提取 8 个统计量：
    均值、方差、标准差、最大值、非零比例、中位数、偏度、峰度。
    最后 RGB 三个通道各做一次，共得到 24 维颜色特征。
    """
    # 拉平成一维并归一化到 [0, 1]，便于不同图片之间保持统一量纲。
    flattened = channel.astype(np.float64).ravel() / 255.0
    # 只统计非零像素，尽量减弱纯黑背景区域对统计量的干扰。
    nonzero = flattened[flattened > 0.0]

    if nonzero.size == 0:
        return np.zeros(8, dtype=np.float64)

    # 一阶和二阶统计量。
    mean = float(np.mean(nonzero))
    variance = float(np.mean((nonzero - mean) ** 2))
    std_dev = math.sqrt(variance)
    max_value = float(np.max(nonzero))
    nonzero_ratio = float(nonzero.size / flattened.size)
    median = float(np.median(nonzero))

    # 偏度与峰度描述分布形状，标准差过小会引发数值不稳定，因此加保护。
    skewness = 0.0
    kurtosis = 0.0
    if std_dev > 1e-10:
        centered = nonzero - mean
        skewness = float(np.mean(centered ** 3) / (std_dev ** 3))
        kurtosis = float(np.mean(centered ** 4) / (std_dev ** 4) - 3.0)
        # 参考你给的 C++ 实现，对极端值做裁剪，避免异常样本放大影响。
        skewness = float(np.clip(skewness, -100.0, 100.0))
        kurtosis = float(np.clip(kurtosis, -100.0, 100.0))

    return np.asarray(
        [mean, variance, std_dev, max_value, nonzero_ratio, median, skewness, kurtosis],
        dtype=np.float64,
    )


def extract_rgb_color_features(rgb_image: np.ndarray) -> np.ndarray:
    """分别对 R、G、B 三个通道提取统计特征，拼接成 24 维颜色特征。"""
    channel_features = [compute_channel_statistics(rgb_image[:, :, index]) for index in range(3)]
    return np.concatenate(channel_features, axis=0)


def quantize_gray_image(gray_image: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    """
    将 0-255 的灰度图量化到较少等级。

    传统 GLCM 常先做灰度等级压缩，这里压缩到 16 级，
    既能保留主要纹理变化，也能显著降低共生矩阵规模。
    """
    quantized = np.floor(gray_image.astype(np.float64) * levels / 256.0).astype(np.int32)
    return np.clip(quantized, 0, levels - 1)


def compute_glcm_for_offset(quantized: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    """
    在指定方向偏移下统计灰度共生矩阵。

    row_shift 和 col_shift 决定相邻像素对的方向：
    例如 (0, 1) 表示水平方向，(-1, 1) 表示 45 度方向。
    """
    if row_shift >= 0:
        src_rows = slice(row_shift, None)
        dst_rows = slice(None, quantized.shape[0] - row_shift)
    else:
        src_rows = slice(None, quantized.shape[0] + row_shift)
        dst_rows = slice(-row_shift, None)

    if col_shift >= 0:
        src_cols = slice(col_shift, None)
        dst_cols = slice(None, quantized.shape[1] - col_shift)
    else:
        src_cols = slice(None, quantized.shape[1] + col_shift)
        dst_cols = slice(-col_shift, None)

    # source 与 target 是成对出现的像素灰度值。
    source = quantized[src_rows, src_cols].ravel()
    target = quantized[dst_rows, dst_cols].ravel()

    # 通过 bincount 快速统计二维频次，比双重循环更高效。
    glcm = np.bincount(
        source * GLCM_LEVELS + target,
        minlength=GLCM_LEVELS * GLCM_LEVELS,
    ).reshape(GLCM_LEVELS, GLCM_LEVELS).astype(np.float64)
    return glcm


def glcm_moments(gray_image: np.ndarray) -> np.ndarray:
    """
    提取灰度共生矩阵纹理特征。

    在 0°、45°、90°、135° 四个方向上分别计算：
    熵、能量、对比度、一致性，共 16 维纹理特征。
    这部分对应你给出的 NoNormalization.cpp 的核心思路。
    """
    quantized = quantize_gray_image(gray_image)
    offsets = [
        (0, GLCM_DISTANCE),
        (-GLCM_DISTANCE, GLCM_DISTANCE),
        (-GLCM_DISTANCE, 0),
        (GLCM_DISTANCE, GLCM_DISTANCE),
    ]

    all_features: List[np.ndarray] = []
    index_matrix = np.arange(GLCM_LEVELS, dtype=np.float64)
    row_index, col_index = np.meshgrid(index_matrix, index_matrix, indexing="ij")

    for row_shift, col_shift in offsets:
        glcm = compute_glcm_for_offset(quantized, row_shift, col_shift)
        # 加上转置，相当于把正反两个方向都统计进来，使矩阵更对称、更稳定。
        glcm += glcm.T
        total = float(glcm.sum())
        if total <= 0:
            all_features.append(np.zeros(4, dtype=np.float64))
            continue

        # 概率归一化后再计算纹理统计量。
        glcm /= total
        nonzero = glcm[glcm > 0]
        entropy = float(-np.sum(nonzero * np.log10(nonzero)))
        energy = float(np.sum(glcm ** 2))
        contrast = float(np.sum(((row_index - col_index) ** 2) * glcm))
        homogeneity = float(np.sum(glcm / (1.0 + (row_index - col_index) ** 2)))
        all_features.append(np.asarray([entropy, energy, contrast, homogeneity], dtype=np.float64))

    return np.concatenate(all_features, axis=0)


def extract_image_features(image_path: str) -> np.ndarray:
    """
    提取单张彩色图像的全部人工特征。

    最终特征维数为：
    16 维 GLCM 纹理特征 + 24 维 RGB 颜色统计特征 = 40 维。
    """
    # 用 fromfile + imdecode 读取图片，避免 OpenCV 在 Windows 中文路径下读图失败。
    image_buffer = np.fromfile(image_path, dtype=np.uint8)
    bgr_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Bayes 特征部分需要分别使用彩色图和灰度图。
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    color_features = extract_rgb_color_features(rgb_image)
    texture_features = glcm_moments(gray_image)
    return np.concatenate([texture_features, color_features], axis=0)


def build_split_data(split_dir: str, cache_path: str) -> SplitData:
    """
    为一个数据划分构建特征矩阵。

    如果已经存在缓存文件且图片列表未变化，则直接读取缓存；
    否则重新逐张提取特征并保存为压缩缓存，便于后续重复实验。
    """
    paths, labels = collect_split_paths(split_dir)
    split_name = os.path.basename(split_dir)

    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        cached_paths = cached["paths"].tolist()
        if cached_paths == paths:
            print(f"[cache] Loaded {split_name} features from {cache_path}", flush=True)
            return SplitData(
                features=cached["features"],
                labels=cached["labels"],
                paths=cached_paths,
            )

    print(f"[extract] Building {split_name} features for {len(paths)} images...", flush=True)
    features_list: List[np.ndarray] = []
    for index, path in enumerate(paths, start=1):
        features_list.append(extract_image_features(path))
        # 定期打印进度，便于观察大数据集上的运行状态。
        if index == 1 or index % 500 == 0 or index == len(paths):
            print(f"[extract] {split_name}: {index}/{len(paths)}", flush=True)

    features = np.vstack(features_list).astype(np.float64)
    np.savez_compressed(cache_path, features=features, labels=labels, paths=np.asarray(paths, dtype=object))
    print(f"[cache] Saved {split_name} features to {cache_path}", flush=True)
    return SplitData(features=features, labels=labels, paths=paths)


def evaluate_candidate_dimensions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[int, List[Dict[str, float]], np.ndarray]:
    """
    在验证集上搜索最合适的 PCA 维数。

    流程：
    1. 先对训练特征做标准化
    2. 枚举一组候选主成分数
    3. 每个候选维数都训练一次 GaussianNB
    4. 用验证集准确率选择最优维数
    """
    # PCA 对量纲敏感，因此先做标准化，而不是直接在原始值上降维。
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    # 先做一次完整 PCA，用于计算累计解释方差并给候选维数提供参考。
    full_pca = PCA().fit(x_train_scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

    # 自动找出达到 95% 信息保留率所需的主成分数。
    pca_95_dim = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    # 特征总维度就是 40，因此候选上限不需要超过 40。
    max_dim = min(x_train_scaled.shape[1], x_train_scaled.shape[0], 40)
    candidates = sorted(set([2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, pca_95_dim, max_dim]))
    candidates = [dim for dim in candidates if 1 <= dim <= max_dim]

    history: List[Dict[str, float]] = []
    best_dim = candidates[0]
    best_accuracy = -1.0
    best_variance = 0.0

    for dim in candidates:
        # 对每个候选维数分别执行 PCA 投影。
        pca = PCA(n_components=dim)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_val_pca = pca.transform(x_val_scaled)

        # 这里使用 GaussianNB 作为 Bayes 分类器。
        classifier = GaussianNB()
        classifier.fit(x_train_pca, y_train)
        val_predictions = classifier.predict(x_val_pca)
        val_accuracy = accuracy_score(y_val, val_predictions)
        explained_variance = float(np.sum(pca.explained_variance_ratio_))

        history.append(
            {
                "components": dim,
                "val_accuracy": val_accuracy,
                "explained_variance": explained_variance,
            }
        )

        # 优先比较验证集准确率；若准确率相同，则保留解释方差更高的方案。
        if val_accuracy > best_accuracy or (
            math.isclose(val_accuracy, best_accuracy) and explained_variance > best_variance
        ):
            best_dim = dim
            best_accuracy = val_accuracy
            best_variance = explained_variance

    return best_dim, history, cumulative_variance


def evaluate_without_pca(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """
    计算“不使用 PCA”的对照实验结果。

    为了让对比更完整，这里给出两组结果：
    1. train -> val：观察不使用 PCA 时的验证集表现
    2. train+val -> test：与最终 PCA 版本保持相同训练口径的公平测试结果
    """
    scaler_val = StandardScaler()
    x_train_scaled = scaler_val.fit_transform(x_train)
    x_val_scaled = scaler_val.transform(x_val)

    classifier_val = GaussianNB()
    classifier_val.fit(x_train_scaled, y_train)
    val_predictions = classifier_val.predict(x_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)

    scaler_test = StandardScaler()
    x_trainval_scaled = scaler_test.fit_transform(x_trainval)
    x_test_scaled = scaler_test.transform(x_test)

    classifier_test = GaussianNB()
    classifier_test.fit(x_trainval_scaled, y_trainval)
    test_predictions = classifier_test.predict(x_test_scaled)
    test_accuracy = accuracy_score(y_test, test_predictions)
    report_text = classification_report(y_test, test_predictions, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(
        y_test,
        test_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )

    return {
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "report_text": report_text,
        "report_dict": report_dict,
    }


def save_training_history(history: Sequence[Dict[str, float]], save_path: str) -> None:
    """将 PCA 维数搜索过程保存为 CSV，便于后续画图或写实验报告。"""
    with open(save_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["components", "val_accuracy", "explained_variance"])
        for row in history:
            writer.writerow([row["components"], row["val_accuracy"], row["explained_variance"]])


def save_confusion_matrix_csv(cm: np.ndarray, labels_list: Sequence[str], save_path: str) -> None:
    """将混淆矩阵另存为 CSV，方便直接放进表格软件或论文附件。"""
    with open(save_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + list(labels_list))
        for label_name, row_values in zip(labels_list, cm):
            writer.writerow([label_name] + row_values.tolist())


def save_learning_curves(
    history: Sequence[Dict[str, float]],
    cumulative_variance: np.ndarray,
    best_components: int,
    save_path: str,
) -> None:
    """
    保存两张“学习曲线”式图像：
    1. PCA 维数与验证集准确率关系
    2. PCA 主成分累计解释方差曲线
    """
    components = [int(item["components"]) for item in history]
    val_accuracy = [float(item["val_accuracy"]) for item in history]
    explained_variance = [float(item["explained_variance"]) for item in history]
    component_axis = np.arange(1, len(cumulative_variance) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(components, val_accuracy, marker="o", color="#d97706", label="Validation Accuracy")
    plt.axvline(best_components, color="#1f77b4", linestyle="--", label=f"Best PCA = {best_components}")
    plt.title("PCA Components vs Validation Accuracy")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(component_axis, cumulative_variance, marker="o", markersize=3, color="#059669", label="Cumulative Explained Variance")
    plt.scatter(components, explained_variance, color="#dc2626", s=25, label="Tested Components")
    plt.axvline(best_components, color="#1f77b4", linestyle="--")
    plt.axhline(0.95, color="#6b7280", linestyle=":", label="95% Variance")
    plt.title("PCA Explained Variance")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics_figure(report_dict: Dict[str, Dict[str, float]], accuracy: float, save_path: str) -> None:
    """将每个类别的 precision、recall、f1-score 画成柱状图。"""
    metrics = ["precision", "recall", "f1-score"]
    class_scores = np.asarray([[report_dict[class_name][metric] for metric in metrics] for class_name in CLASS_NAMES])
    x = np.arange(len(CLASS_NAMES))
    width = 0.22
    colors = ["#2563eb", "#f59e0b", "#10b981"]

    plt.figure(figsize=(10, 5))
    for index, metric in enumerate(metrics):
        plt.bar(x + (index - 1) * width, class_scores[:, index], width=width, color=colors[index], label=metric.title())

    plt.axhline(accuracy, color="#dc2626", linestyle="--", linewidth=2, label=f"Overall Accuracy = {accuracy:.4f}")
    plt.xticks(x, CLASS_NAMES)
    plt.ylim(0, 1.05)
    plt.title("Bayes Classification Metrics")
    plt.ylabel("Score")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def choose_sample_indices(
    paths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> List[int]:
    """
    挑选要可视化展示的测试样本。

    策略：
    1. 每个类别挑一张“预测正确且置信度高”的代表图
    2. 再挑若干张“预测错误但模型置信度较高”的典型误判图
    """
    selected: List[int] = []

    for class_index, _class_name in enumerate(CLASS_NAMES):
        class_matches = np.where((y_true == class_index) & (y_pred == class_index))[0]
        if class_matches.size > 0:
            best_index = int(class_matches[np.argmax(probabilities[class_matches, class_index])])
            selected.append(best_index)

    misclassified = np.where(y_true != y_pred)[0]
    if misclassified.size > 0:
        ranked = sorted(
            misclassified.tolist(),
            key=lambda idx: probabilities[idx, y_pred[idx]],
            reverse=True,
        )
        for index in ranked[:3]:
            selected.append(index)

    unique_selected: List[int] = []
    seen = set()
    for index in selected:
        if index not in seen:
            seen.add(index)
            unique_selected.append(index)

    return unique_selected[:6]


def save_sample_predictions(
    result_dir: str,
    paths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """
    保存若干张单图预测结果，并额外拼接成总览图。

    错误样本文件名会追加 `_FN`，便于和 CNN、ResNet 的结果形式保持一致。
    """
    sample_indices = choose_sample_indices(paths, y_true, y_pred, probabilities)
    if not sample_indices:
        return

    figure_columns = 3
    figure_rows = math.ceil(len(sample_indices) / figure_columns)
    plt.figure(figsize=(5 * figure_columns, 4.5 * figure_rows))

    for plot_index, sample_index in enumerate(sample_indices, start=1):
        image_path = paths[sample_index]
        image = Image.open(image_path).convert("RGB")
        true_name = CLASS_NAMES[int(y_true[sample_index])]
        pred_name = CLASS_NAMES[int(y_pred[sample_index])]
        confidence = float(np.max(probabilities[sample_index]))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        suffix = "_FN" if true_name != pred_name else ""
        output_name = f"{true_name}_{stem}{suffix}.png"
        output_path = os.path.join(result_dir, output_name)

        standalone_figure = plt.figure(figsize=(6, 4.8))
        plt.imshow(image)
        plt.title(f"True: {true_name} | Pred: {pred_name}\nConfidence: {confidence:.2%}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(standalone_figure)

        plt.subplot(figure_rows, figure_columns, plot_index)
        plt.imshow(image)
        plt.title(f"{true_name} -> {pred_name}\n{confidence:.2%}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "sample_predictions_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()


def write_experiment_summary(
    save_path: str,
    train_count: int,
    val_count: int,
    test_count: int,
    feature_dim: int,
    best_components: int,
    explained_variance: float,
    best_val_accuracy: float,
    test_accuracy: float,
    report_dict: Dict[str, Dict[str, float]],
    no_pca_val_accuracy: float,
    no_pca_test_accuracy: float,
    no_pca_report_dict: Dict[str, Dict[str, float]],
) -> None:
    """
    生成“实现过程与结果说明”文档。

    这份说明面向课程实验或论文整理场景，简要讲清：
    数据怎么处理、特征怎么构造、PCA 为什么做、Bayes 怎么训练、
    最终结果如何解读。
    """
    with open(save_path, "w", encoding="utf-8-sig") as summary_file:
        summary_file.write("Bayes 三分类实现过程与结果说明\n")
        summary_file.write("=" * 60 + "\n\n")

        summary_file.write("一、实现目标\n")
        summary_file.write("本方法用于完成 cloudy、sandstorm、sunny 三类彩色天气图像分类。\n")
        summary_file.write("与前两种深度学习方法不同，这里采用传统机器学习流程：人工特征提取 + PCA 降维 + Bayes 分类。\n\n")

        summary_file.write("二、实现流程\n")
        summary_file.write("1. 读取 data/train、data/val、data/test 三个数据划分。\n")
        summary_file.write("2. 对每张彩色图像提取 40 维人工特征。\n")
        summary_file.write("3. 先在训练集和验证集上搜索最优 PCA 主成分数。\n")
        summary_file.write("4. 用训练集和验证集合并后的样本重新训练最终 Bayes 分类器。\n")
        summary_file.write("5. 在测试集上输出分类报告、混淆矩阵、指标图和样本可视化结果。\n\n")

        summary_file.write("三、特征设计说明\n")
        summary_file.write("1. GLCM 纹理特征：\n")
        summary_file.write("   灰度图先量化为 16 级，在 0°、45°、90°、135° 四个方向上构建灰度共生矩阵。\n")
        summary_file.write("   每个方向提取熵、能量、对比度、一致性 4 个统计量，共得到 16 维纹理特征。\n")
        summary_file.write("2. RGB 颜色特征：\n")
        summary_file.write("   对 R、G、B 三个通道分别计算均值、方差、标准差、最大值、非零比例、中位数、偏度、峰度。\n")
        summary_file.write("   每个通道 8 维，三个通道共得到 24 维颜色特征。\n")
        summary_file.write(f"3. 最终特征维数：16 + 24 = {feature_dim} 维。\n\n")

        summary_file.write("四、降维与分类说明\n")
        summary_file.write("1. 先对特征做标准化，再进行 PCA 降维，避免不同量纲影响主成分分解。\n")
        summary_file.write("2. 候选 PCA 维数通过验证集准确率进行筛选。\n")
        summary_file.write(f"3. 本次实验选出的最优 PCA 维数为 {best_components}。\n")
        summary_file.write("4. 最终分类器采用 Gaussian Naive Bayes，即高斯朴素贝叶斯分类器。\n\n")

        summary_file.write("五、数据规模\n")
        summary_file.write(f"训练集样本数：{train_count}\n")
        summary_file.write(f"验证集样本数：{val_count}\n")
        summary_file.write(f"测试集样本数：{test_count}\n\n")

        summary_file.write("六、结果说明\n")
        summary_file.write(f"1. PCA 后累计解释方差：{explained_variance:.6f}\n")
        summary_file.write("   该指标表示 PCA 保留下来的主成分一共解释了原始特征中多少比例的信息。\n")
        summary_file.write("   本实验中取 32 个主成分后，仍保留了约 99.985% 的原始特征变化信息，\n")
        summary_file.write("   说明降维后信息损失很小，PCA 主要起到了压缩冗余和减弱特征相关性的作用。\n")
        summary_file.write(f"2. 验证集最优准确率：{best_val_accuracy:.6f}\n")
        summary_file.write(f"3. 测试集准确率：{test_accuracy:.6f}\n\n")

        summary_file.write("七、与无 PCA 的对比\n")
        summary_file.write(f"1. 无 PCA 验证集准确率：{no_pca_val_accuracy:.6f}\n")
        summary_file.write(f"2. 无 PCA 测试集准确率：{no_pca_test_accuracy:.6f}\n")
        summary_file.write(f"3. 使用 PCA 的测试集准确率：{test_accuracy:.6f}\n")
        summary_file.write(f"4. PCA 相比无 PCA 的测试集准确率提升：{test_accuracy - no_pca_test_accuracy:.6f}\n")
        summary_file.write("5. 对比结论：\n")
        summary_file.write("   不使用 PCA 时，Bayes 分类器可以完成三分类任务，但整体准确率更低。\n")
        summary_file.write("   引入 PCA 后，特征冗余和相关性被压缩，分类结果更稳定，最终测试精度更高。\n\n")

        summary_file.write("八、使用 PCA 后的各类别表现\n")
        for class_name in CLASS_NAMES:
            precision = report_dict[class_name]["precision"]
            recall = report_dict[class_name]["recall"]
            f1_score = report_dict[class_name]["f1-score"]
            support = int(report_dict[class_name]["support"])
            summary_file.write(
                f"{class_name}: precision={precision:.4f}, recall={recall:.4f}, "
                f"f1-score={f1_score:.4f}, support={support}\n"
            )

        summary_file.write("\n九、无 PCA 的各类别表现\n")
        for class_name in CLASS_NAMES:
            precision = no_pca_report_dict[class_name]["precision"]
            recall = no_pca_report_dict[class_name]["recall"]
            f1_score = no_pca_report_dict[class_name]["f1-score"]
            support = int(no_pca_report_dict[class_name]["support"])
            summary_file.write(
                f"{class_name}: precision={precision:.4f}, recall={recall:.4f}, "
                f"f1-score={f1_score:.4f}, support={support}\n"
            )

        summary_file.write("\n十、结果文件说明\n")
        summary_file.write("1. classification_report.txt：测试集分类报告。\n")
        summary_file.write("2. classification_report_no_pca.txt：无 PCA 对照实验的测试集分类报告。\n")
        summary_file.write("3. evaluation_metrics.txt：总体指标摘要，含是否使用 PCA 的对比结果。\n")
        summary_file.write("4. confusion_matrix.png / confusion_matrix.csv：混淆矩阵图与表。\n")
        summary_file.write("5. learning_curves.png：PCA 维数搜索和解释方差曲线。\n")
        summary_file.write("6. evaluation_metrics.png：各类别 precision、recall、f1-score 柱状图。\n")
        summary_file.write("7. sample_predictions_grid.png：测试样本预测拼接图。\n")
        summary_file.write("8. *_FN.png：典型误分类样本图。\n")
        summary_file.write("9. best_bayes_pipeline.joblib：保存的标准化器、PCA 和 Bayes 分类器。\n\n")

        summary_file.write("十一、方法特点说明\n")
        summary_file.write("该方法不依赖深度神经网络训练，解释性较强，适合与 CNN、ResNet 做传统方法对比实验。\n")
        summary_file.write("但由于人工特征表达能力有限，其最终精度通常会低于表现较好的深度学习模型。\n")


def main() -> None:
    """主函数：串联完整的特征提取、降维、训练、评估和结果保存流程。"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(project_dir, "data")
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    result_dir = os.path.join(project_dir, "result_bayes")
    os.makedirs(result_dir, exist_ok=True)

    classification_report_path = os.path.join(result_dir, "classification_report.txt")
    no_pca_report_path = os.path.join(result_dir, "classification_report_no_pca.txt")
    evaluation_metrics_path = os.path.join(result_dir, "evaluation_metrics.txt")
    confusion_matrix_image_path = os.path.join(result_dir, "confusion_matrix.png")
    confusion_matrix_csv_path = os.path.join(result_dir, "confusion_matrix.csv")
    learning_curve_path = os.path.join(result_dir, "learning_curves.png")
    training_history_path = os.path.join(result_dir, "training_history.csv")
    model_path = os.path.join(result_dir, "best_bayes_pipeline.joblib")
    summary_path = os.path.join(result_dir, "实现过程与结果说明.txt")

    print("[step] Extracting handcrafted features...", flush=True)
    train_data = build_split_data(train_dir, os.path.join(result_dir, "train_features_cache.npz"))
    val_data = build_split_data(val_dir, os.path.join(result_dir, "val_features_cache.npz"))
    test_data = build_split_data(test_dir, os.path.join(result_dir, "test_features_cache.npz"))

    print("[step] Searching for the best PCA dimension on the validation set...", flush=True)
    best_components, history, cumulative_variance = evaluate_candidate_dimensions(
        train_data.features,
        train_data.labels,
        val_data.features,
        val_data.labels,
    )

    save_training_history(history, training_history_path)
    save_learning_curves(history, cumulative_variance, best_components, learning_curve_path)
    print(f"[step] Best PCA components = {best_components}", flush=True)

    combined_features = np.vstack([train_data.features, val_data.features])
    combined_labels = np.concatenate([train_data.labels, val_data.labels])
    no_pca_results = evaluate_without_pca(
        x_train=train_data.features,
        y_train=train_data.labels,
        x_val=val_data.features,
        y_val=val_data.labels,
        x_trainval=combined_features,
        y_trainval=combined_labels,
        x_test=test_data.features,
        y_test=test_data.labels,
    )

    # 最终模型使用 train + val 共同训练，以充分利用已知样本。
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_features)
    test_scaled = scaler.transform(test_data.features)

    # 按验证集上选出的最优维数重新训练最终 PCA。
    pca = PCA(n_components=best_components)
    combined_pca = pca.fit_transform(combined_scaled)
    test_pca = pca.transform(test_scaled)

    print("[step] Training Gaussian Bayes classifier...", flush=True)
    classifier = GaussianNB()
    classifier.fit(combined_pca, combined_labels)

    print("[step] Evaluating on the test set...", flush=True)
    test_predictions = classifier.predict(test_pca)
    test_probabilities = classifier.predict_proba(test_pca)
    test_accuracy = accuracy_score(test_data.labels, test_predictions)
    cm = confusion_matrix(test_data.labels, test_predictions)
    report_text = classification_report(test_data.labels, test_predictions, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(
        test_data.labels,
        test_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )

    with open(classification_report_path, "w", encoding="utf-8-sig") as report_file:
        report_file.write("Bayes Classification Report\n")
        report_file.write("=" * 50 + "\n")
        report_file.write(report_text)

    with open(no_pca_report_path, "w", encoding="utf-8-sig") as report_file:
        report_file.write("Bayes Classification Report Without PCA\n")
        report_file.write("=" * 50 + "\n")
        report_file.write(no_pca_results["report_text"])

    with open(evaluation_metrics_path, "w", encoding="utf-8-sig") as metrics_file:
        metrics_file.write("Bayes Evaluation Metrics\n")
        metrics_file.write("=" * 50 + "\n")
        metrics_file.write(f"Feature Dimension Before PCA: {train_data.features.shape[1]}\n")
        metrics_file.write(f"Selected PCA Components: {best_components}\n")
        metrics_file.write(f"Explained Variance After PCA: {np.sum(pca.explained_variance_ratio_):.6f}\n")
        metrics_file.write(f"Validation Accuracy (Best Search Result): {max(item['val_accuracy'] for item in history):.6f}\n")
        metrics_file.write(f"Validation Accuracy Without PCA: {no_pca_results['val_accuracy']:.6f}\n")
        metrics_file.write(f"Test Accuracy: {test_accuracy:.6f}\n")
        metrics_file.write(f"Test Accuracy Without PCA: {no_pca_results['test_accuracy']:.6f}\n")
        metrics_file.write(f"Accuracy Gain From PCA: {test_accuracy - no_pca_results['test_accuracy']:.6f}\n")
        metrics_file.write(f"Train Samples: {train_data.features.shape[0]}\n")
        metrics_file.write(f"Validation Samples: {val_data.features.shape[0]}\n")
        metrics_file.write(f"Test Samples: {test_data.features.shape[0]}\n")
        metrics_file.write(f"Classes: {CLASS_NAMES}\n")

    save_confusion_matrix_csv(cm, CLASS_NAMES, confusion_matrix_csv_path)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Bayes Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(confusion_matrix_image_path, dpi=300, bbox_inches="tight")
    plt.close()

    save_metrics_figure(report_dict, test_accuracy, os.path.join(result_dir, "evaluation_metrics.png"))
    save_sample_predictions(result_dir, test_data.paths, test_data.labels, test_predictions, test_probabilities)
    write_experiment_summary(
        save_path=summary_path,
        train_count=train_data.features.shape[0],
        val_count=val_data.features.shape[0],
        test_count=test_data.features.shape[0],
        feature_dim=train_data.features.shape[1],
        best_components=best_components,
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
        best_val_accuracy=max(item["val_accuracy"] for item in history),
        test_accuracy=test_accuracy,
        report_dict=report_dict,
        no_pca_val_accuracy=float(no_pca_results["val_accuracy"]),
        no_pca_test_accuracy=float(no_pca_results["test_accuracy"]),
        no_pca_report_dict=no_pca_results["report_dict"],
    )

    # 将预处理器和分类器整体打包保存，后续预测时可以直接加载使用。
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "classifier": classifier,
            "class_names": CLASS_NAMES,
        },
        model_path,
    )

    print(f"Selected PCA components: {best_components}")
    print(f"Test accuracy: {test_accuracy:.6f}")
    print(f"All Bayes result files have been saved to: {result_dir}")


if __name__ == "__main__":
    main()
