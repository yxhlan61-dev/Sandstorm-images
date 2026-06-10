import csv
import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

CLASS_NAMES = ["cloudy", "sandstorm", "sunny"]


def read_history_csv(csv_path: str) -> List[Dict[str, float]]:
    """读取训练历史 CSV，便于在说明文件中引用关键训练结果。"""
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: List[Dict[str, float]] = []
        for row in reader:
            parsed = {key: float(value) if key != "epoch" else int(value) for key, value in row.items()}
            rows.append(parsed)
    return rows


def dataset_counts(base_dir: str) -> Dict[str, Dict[str, int]]:
    """统计 train / val / test 中每个类别的样本数量。"""
    counts: Dict[str, Dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_counts: Dict[str, int] = {}
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(base_dir, split, class_name)
            split_counts[class_name] = len(
                [name for name in os.listdir(class_dir) if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            )
        counts[split] = split_counts
    return counts


def report_to_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """生成字典形式的分类报告，方便写入说明文件。"""
    return classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )


def save_confusion_matrix_csv(cm: np.ndarray, save_path: str) -> None:
    """将混淆矩阵保存为 CSV。"""
    with open(save_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + CLASS_NAMES)
        for class_name, row in zip(CLASS_NAMES, cm):
            writer.writerow([class_name] + row.tolist())


def save_confusion_matrix_image(cm: np.ndarray, save_path: str, title: str) -> None:
    """将混淆矩阵保存为 PNG。"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics_figure(report_dict: Dict[str, Dict[str, float]], accuracy: float, save_path: str, title: str) -> None:
    """为 CNN 额外生成一张指标柱状图，使结果形式更完整。"""
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
    plt.title(title)
    plt.ylabel("Score")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_cnn_test_generator(base_dir: str):
    """构建与 CNN 训练脚本一致的测试集生成器。"""
    test_dir = os.path.join(base_dir, "test")
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )


def evaluate_cnn(project_dir: str) -> Dict[str, object]:
    """读取已保存的 CNN 模型，在测试集上补做一次规范评估。"""
    base_dir = os.path.join(project_dir, "data")
    result_dir = os.path.join(project_dir, "result_CNN")
    model_path = os.path.join(result_dir, "optimized_sandstorm_model.h5")

    test_generator = build_cnn_test_generator(base_dir)
    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    test_generator.reset()
    probabilities = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_generator.classes
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_dict = report_to_dict(y_true, y_pred)

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
        "report_text": report_text,
        "report_dict": report_dict,
    }


def write_cnn_result_files(project_dir: str, cnn_metrics: Dict[str, object], counts: Dict[str, Dict[str, int]]) -> None:
    """把 CNN 的补充评估结果保存到 result_CNN。"""
    result_dir = os.path.join(project_dir, "result_CNN")
    classification_report_path = os.path.join(result_dir, "classification_report.txt")
    evaluation_metrics_path = os.path.join(result_dir, "evaluation_metrics.txt")
    confusion_matrix_csv_path = os.path.join(result_dir, "confusion_matrix.csv")
    confusion_matrix_png_path = os.path.join(result_dir, "confusion_matrix.png")
    metrics_figure_path = os.path.join(result_dir, "evaluation_metrics_detailed.png")

    with open(classification_report_path, "w", encoding="utf-8-sig") as report_file:
        report_file.write("CNN Classification Report\n")
        report_file.write("=" * 50 + "\n")
        report_file.write(cnn_metrics["report_text"])

    with open(evaluation_metrics_path, "w", encoding="utf-8-sig") as metrics_file:
        metrics_file.write("CNN Evaluation Metrics\n")
        metrics_file.write("=" * 50 + "\n")
        metrics_file.write(f"Test Loss: {cnn_metrics['test_loss']:.6f}\n")
        metrics_file.write(f"Test Accuracy: {cnn_metrics['test_accuracy']:.6f}\n")
        metrics_file.write(f"Train Samples: {sum(counts['train'].values())}\n")
        metrics_file.write(f"Validation Samples: {sum(counts['val'].values())}\n")
        metrics_file.write(f"Test Samples: {sum(counts['test'].values())}\n")
        metrics_file.write(f"Classes: {CLASS_NAMES}\n")

    save_confusion_matrix_csv(cnn_metrics["cm"], confusion_matrix_csv_path)
    save_confusion_matrix_image(cnn_metrics["cm"], confusion_matrix_png_path, "CNN Confusion Matrix")
    save_metrics_figure(
        cnn_metrics["report_dict"],
        cnn_metrics["test_accuracy"],
        metrics_figure_path,
        "CNN Classification Metrics",
    )


def write_cnn_summary(project_dir: str, cnn_metrics: Dict[str, object], counts: Dict[str, Dict[str, int]]) -> None:
    """生成 CNN 的实现过程与结果说明。"""
    result_dir = os.path.join(project_dir, "result_CNN")
    summary_path = os.path.join(result_dir, "实现过程与结果说明.txt")

    with open(summary_path, "w", encoding="utf-8-sig") as summary_file:
        summary_file.write("CNN 三分类实现过程与结果说明\n")
        summary_file.write("=" * 60 + "\n\n")

        summary_file.write("一、实现目标\n")
        summary_file.write("本方法使用卷积神经网络完成 cloudy、sandstorm、sunny 三类彩色天气图像分类。\n")
        summary_file.write("相对于传统人工特征方法，CNN 通过端到端训练自动学习图像的边缘、纹理、形状和高级语义特征。\n\n")

        summary_file.write("二、实现流程\n")
        summary_file.write("1. 使用 data/train、data/val、data/test 三个数据划分。\n")
        summary_file.write("2. 对训练集进行数据增强，对验证集和测试集仅做归一化。\n")
        summary_file.write("3. 构建多层卷积神经网络并训练。\n")
        summary_file.write("4. 在测试集上输出分类报告、混淆矩阵和样本可视化结果。\n\n")

        summary_file.write("三、模型结构说明\n")
        summary_file.write("1. 输入尺寸为 150×150×3。\n")
        summary_file.write("2. 主体由 4 个卷积块组成，每个卷积块包含 Conv2D + BatchNormalization + MaxPooling。\n")
        summary_file.write("3. 卷积通道数依次为 32、64、128、256。\n")
        summary_file.write("4. 特征图展平后接 512 维全连接层，并使用 Dropout(0.6) 抑制过拟合。\n")
        summary_file.write("5. 输出层使用 softmax 完成三分类。\n\n")

        summary_file.write("四、训练策略说明\n")
        summary_file.write("1. 训练集采用旋转、平移、剪切、缩放、水平翻转等数据增强。\n")
        summary_file.write("2. 优化器采用 Adam，学习率为 0.0005。\n")
        summary_file.write("3. 损失函数为 categorical_crossentropy，评价指标为 accuracy。\n")
        summary_file.write("4. 为了减少漏判 sandstorm，对 sandstorm 类设置了更高的类别权重。\n")
        summary_file.write("5. 使用 EarlyStopping，根据验证集 loss 自动提前停止训练。\n\n")

        summary_file.write("五、数据规模\n")
        summary_file.write(f"训练集样本数：{sum(counts['train'].values())}\n")
        summary_file.write(f"验证集样本数：{sum(counts['val'].values())}\n")
        summary_file.write(f"测试集样本数：{sum(counts['test'].values())}\n")
        for split in ["train", "val", "test"]:
            summary_file.write(
                f"{split} 细分：cloudy={counts[split]['cloudy']}, "
                f"sandstorm={counts[split]['sandstorm']}, sunny={counts[split]['sunny']}\n"
            )
        summary_file.write("\n")

        summary_file.write("六、结果说明\n")
        summary_file.write(f"1. 测试集损失值：{cnn_metrics['test_loss']:.6f}\n")
        summary_file.write(f"2. 测试集准确率：{cnn_metrics['test_accuracy']:.6f}\n")
        summary_file.write("3. 该准确率表示测试集中被正确分类的样本占总样本的比例。\n")
        summary_file.write("4. CNN 通过自动学习图像特征，相比传统手工特征方法通常具备更强的表达能力。\n\n")

        summary_file.write("七、各类别表现\n")
        for class_name in CLASS_NAMES:
            class_metrics = cnn_metrics["report_dict"][class_name]
            summary_file.write(
                f"{class_name}: precision={class_metrics['precision']:.4f}, "
                f"recall={class_metrics['recall']:.4f}, "
                f"f1-score={class_metrics['f1-score']:.4f}, "
                f"support={int(class_metrics['support'])}\n"
            )
        summary_file.write("\n")

        summary_file.write("八、结果文件说明\n")
        summary_file.write("1. classification_report.txt：CNN 测试集分类报告。\n")
        summary_file.write("2. evaluation_metrics.txt：CNN 总体指标摘要。\n")
        summary_file.write("3. confusion_matrix.png / confusion_matrix.csv：CNN 混淆矩阵图与表。\n")
        summary_file.write("4. evaluation_metrics_detailed.png：CNN 各类别指标柱状图。\n")
        summary_file.write("5. 准确率与损失值变化图.png、学习曲线改进版.png：训练过程曲线图。\n")
        summary_file.write("6. 神经网络结构图.png：网络结构可视化结果。\n")
        summary_file.write("7. 拼接图.png 与若干单张图片：测试样本预测可视化结果。\n")
        summary_file.write("8. optimized_sandstorm_model.h5：训练完成后的 CNN 模型文件。\n\n")

        summary_file.write("九、方法特点说明\n")
        summary_file.write("CNN 能直接从像素中学习判别特征，结构清晰、训练逻辑直观，适合作为基础深度学习方法进行对比实验。\n")
        summary_file.write("但其性能较大程度上依赖网络结构设计、数据增强质量和训练策略设置。\n")


def parse_metric_file(file_path: str) -> Dict[str, str]:
    """读取 evaluation_metrics.txt 这类 key:value 结构文件。"""
    metrics: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8-sig") as metric_file:
        for line in metric_file:
            line = line.strip()
            if not line or ":" not in line or line.startswith("="):
                continue
            key, value = line.split(":", 1)
            metrics[key.strip()] = value.strip()
    return metrics


def write_resnet_summary(project_dir: str, counts: Dict[str, Dict[str, int]]) -> None:
    """根据现有 ResNet 训练和评估结果生成说明文件。"""
    result_dir = os.path.join(project_dir, "result_resnet")
    summary_path = os.path.join(result_dir, "实现过程与结果说明.txt")
    report_path = os.path.join(result_dir, "classification_report.txt")
    metric_path = os.path.join(result_dir, "evaluation_metrics.txt")
    history_path = os.path.join(result_dir, "training_history.csv")

    history = read_history_csv(history_path)
    metric_map = parse_metric_file(metric_path)
    test_loss = float(metric_map["Test Loss"])
    test_accuracy = float(metric_map["Test Accuracy"])

    best_val_accuracy = max(row["val_accuracy"] for row in history) if history else 0.0
    final_epoch = history[-1]["epoch"] if history else 0

    with open(report_path, "r", encoding="utf-8-sig") as report_file:
        report_text = report_file.read()

    lines = [line for line in report_text.splitlines() if line.strip()]
    class_metrics: Dict[str, Tuple[float, float, float, int]] = {}
    for line in lines:
        parts = line.split()
        if parts and parts[0] in CLASS_NAMES and len(parts) >= 5:
            class_metrics[parts[0]] = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                int(parts[4]),
            )

    with open(summary_path, "w", encoding="utf-8-sig") as summary_file:
        summary_file.write("ResNet50 三分类实现过程与结果说明\n")
        summary_file.write("=" * 60 + "\n\n")

        summary_file.write("一、实现目标\n")
        summary_file.write("本方法使用 ResNet50 迁移学习完成 cloudy、sandstorm、sunny 三类彩色天气图像分类。\n")
        summary_file.write("相对于从零开始训练的普通 CNN，ResNet50 利用了 ImageNet 预训练参数，能够更高效地学习稳定的图像表示。\n\n")

        summary_file.write("二、实现流程\n")
        summary_file.write("1. 读取 data/train、data/val、data/test 三个数据划分。\n")
        summary_file.write("2. 使用 ResNet50 专用的 preprocess_input 进行输入预处理。\n")
        summary_file.write("3. 第一阶段冻结 ResNet50 主干，仅训练新接入的分类头。\n")
        summary_file.write("4. 第二阶段解冻高层网络，进行小学习率微调。\n")
        summary_file.write("5. 使用验证集最优模型在测试集上输出最终评估结果。\n\n")

        summary_file.write("三、模型结构说明\n")
        summary_file.write("1. 输入尺寸为 224×224×3。\n")
        summary_file.write("2. 主干网络使用 ImageNet 预训练的 ResNet50，include_top=False。\n")
        summary_file.write("3. 分类头由 GlobalAveragePooling2D + Dense(256, relu) + Dropout(0.5) + Dense(3, softmax) 组成。\n")
        summary_file.write("4. 第一阶段冻结主干网络，第二阶段只解冻最后约 30 层进行微调。\n\n")

        summary_file.write("四、训练策略说明\n")
        summary_file.write("1. 训练集采用旋转、平移、剪切、缩放和水平翻转等数据增强。\n")
        summary_file.write("2. 第一阶段学习率为 1e-3，第二阶段微调学习率为 1e-5。\n")
        summary_file.write("3. 使用 ModelCheckpoint 保存验证集最优模型。\n")
        summary_file.write("4. 使用 EarlyStopping 防止过拟合，使用 ReduceLROnPlateau 自动调节学习率。\n")
        summary_file.write("5. 对 sandstorm 类别设置更高类别权重，以提升其识别稳定性。\n\n")

        summary_file.write("五、数据规模\n")
        summary_file.write(f"训练集样本数：{sum(counts['train'].values())}\n")
        summary_file.write(f"验证集样本数：{sum(counts['val'].values())}\n")
        summary_file.write(f"测试集样本数：{sum(counts['test'].values())}\n")
        for split in ["train", "val", "test"]:
            summary_file.write(
                f"{split} 细分：cloudy={counts[split]['cloudy']}, "
                f"sandstorm={counts[split]['sandstorm']}, sunny={counts[split]['sunny']}\n"
            )
        summary_file.write("\n")

        summary_file.write("六、结果说明\n")
        summary_file.write(f"1. 最终测试损失值：{test_loss:.6f}\n")
        summary_file.write(f"2. 最终测试准确率：{test_accuracy:.6f}\n")
        summary_file.write(f"3. 训练过程中最高验证集准确率：{best_val_accuracy:.6f}\n")
        summary_file.write(f"4. 训练历史共记录到第 {final_epoch} 个 epoch。\n")
        summary_file.write("5. 该结果说明迁移学习模型在当前三分类任务上具有较高的泛化能力。\n\n")

        summary_file.write("七、各类别表现\n")
        for class_name in CLASS_NAMES:
            precision, recall, f1_score, support = class_metrics[class_name]
            summary_file.write(
                f"{class_name}: precision={precision:.4f}, recall={recall:.4f}, "
                f"f1-score={f1_score:.4f}, support={support}\n"
            )
        summary_file.write("\n")

        summary_file.write("八、结果文件说明\n")
        summary_file.write("1. classification_report.txt：ResNet50 测试集分类报告。\n")
        summary_file.write("2. evaluation_metrics.txt：ResNet50 总体指标摘要。\n")
        summary_file.write("3. confusion_matrix.png / confusion_matrix.csv：ResNet50 混淆矩阵图与表。\n")
        summary_file.write("4. learning_curves.png：两阶段训练的准确率和损失曲线。\n")
        summary_file.write("5. training_history.csv：逐 epoch 的训练/验证指标记录。\n")
        summary_file.write("6. best_resnet50_sandstorm_model.keras：验证集最优模型。\n")
        summary_file.write("7. optimized_sandstorm_resnet50_model.h5：最终导出的 ResNet50 模型文件。\n\n")

        summary_file.write("九、方法特点说明\n")
        summary_file.write("ResNet50 借助预训练特征和残差结构，在中小规模数据集上通常比普通 CNN 更稳定，收敛更快、精度更高。\n")
        summary_file.write("但其模型体量更大，对显存、训练时间和部署资源的要求也更高。\n")


def main() -> None:
    """生成 CNN 与 ResNet50 的说明文件，并补齐 CNN 的规范评估结果。"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(project_dir, "data")
    counts = dataset_counts(base_dir)

    cnn_metrics = evaluate_cnn(project_dir)
    write_cnn_result_files(project_dir, cnn_metrics, counts)
    write_cnn_summary(project_dir, cnn_metrics, counts)
    write_resnet_summary(project_dir, counts)

    print("Generated CNN and ResNet50 explanation files successfully.")


if __name__ == "__main__":
    main()
