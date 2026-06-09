import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 这份脚本使用 ResNet50 做天气图像三分类：
# cloudy / sandstorm / sunny
#
# 与你之前手写的 CNN 不同，这里使用的是“迁移学习”：
# 1. 先加载一个已经在 ImageNet 大型数据集上训练好的 ResNet50
# 2. 保留它已经学会的通用图像特征提取能力
# 3. 在最后接上适合我们三分类任务的输出层
# 4. 先只训练新加的分类层，再少量微调原始网络的高层
#
# 这样做的好处：
# - 训练速度通常更快
# - 小数据集上通常比从零开始训练更稳定
# - 对初学者来说，也更适合做“模型对比实验”

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

# --- 1. 数据集路径设置 ---
# 这里默认你的数据目录结构仍然是：
# data/
#   train/
#     cloudy/
#     sandstorm/
#     sunny/
#   val/
#     cloudy/
#     sandstorm/
#     sunny/
#   test/
#     cloudy/
#     sandstorm/
#     sunny/
#
# flow_from_directory 会自动把文件夹名当成类别名。
base_dir = r"d:\桌面\science research\Sandstorm Picture\data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# 所有 ResNet 结果统一保存到 result_resnet 文件夹。
project_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(project_dir, "result_resnet")
os.makedirs(result_dir, exist_ok=True)

# 下面这些路径都是“结果文件”的保存位置。
best_model_path = os.path.join(result_dir, "best_resnet50_sandstorm_model.keras")
final_model_path = os.path.join(result_dir, "optimized_sandstorm_resnet50_model.h5")
classification_report_path = os.path.join(result_dir, "classification_report.txt")
evaluation_metrics_path = os.path.join(result_dir, "evaluation_metrics.txt")
confusion_matrix_image_path = os.path.join(result_dir, "confusion_matrix.png")
confusion_matrix_csv_path = os.path.join(result_dir, "confusion_matrix.csv")
learning_curve_path = os.path.join(result_dir, "learning_curves.png")
training_history_path = os.path.join(result_dir, "training_history.csv")

# ResNet50 常用输入尺寸是 224x224。
# 你之前的 CNN 用的是 150x150，这里改成 224x224 是为了更贴近
# 预训练模型原本的设计习惯，一般会更稳定一些。
img_height = 224
img_width = 224
batch_size = 32

# --- 2. 数据预处理和数据增强 ---
# 这里和你之前的 CNN 脚本思路类似，训练集做增强，验证集/测试集不做增强。
#
# 重要区别：
# 之前的 CNN 使用 rescale=1./255，把像素缩放到 [0, 1]
# 这里不能简单使用 rescale，因为 ResNet50 有自己专用的输入预处理方式。
# 所以要使用 preprocess_input。
#
# preprocess_input 的作用：
# - 把图片数组转换成更适合 ResNet50 预训练权重的数值形式
# - 让输入分布和 ImageNet 训练时更接近
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 训练集：做数据增强，帮助模型看到更多“变化后的样本”
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
)

# 验证集：不做随机增强，只做预处理
val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
)

# 测试集：同样不做随机增强，并且必须 shuffle=False
# 这样预测结果的顺序才能和真实标签一一对应，方便后面计算混淆矩阵。
test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

class_indices = train_generator.class_indices
labels = list(class_indices.keys())
num_classes = len(labels)


def save_training_history(history_stage1, history_stage2, save_path):
    """把两个训练阶段的学习曲线指标保存成 CSV，方便后续画图或做表格。"""
    acc = history_stage1.history["accuracy"] + history_stage2.history["accuracy"]
    val_acc = history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"]
    loss = history_stage1.history["loss"] + history_stage2.history["loss"]
    val_loss = history_stage1.history["val_loss"] + history_stage2.history["val_loss"]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_accuracy", "val_accuracy", "train_loss", "val_loss"])
        for epoch_index, (a, va, l, vl) in enumerate(zip(acc, val_acc, loss, val_loss), start=1):
            writer.writerow([epoch_index, a, va, l, vl])

    return acc, val_acc, loss, val_loss


def save_confusion_matrix_csv(cm, labels_list, save_path):
    """把混淆矩阵保存成 CSV，方便后续插入报告或在 Excel 中查看。"""
    with open(save_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + labels_list)
        for label_name, row_values in zip(labels_list, cm):
            writer.writerow([label_name] + row_values.tolist())


# --- 3. 构建 ResNet50 迁移学习模型 ---
# include_top=False 的意思是：
# 不使用 ResNet50 原来为 ImageNet 1000 分类准备的最后输出层，
# 只保留它前面的“特征提取部分”。
#
# weights="imagenet" 表示加载在 ImageNet 数据集上训练好的参数。
# 这是迁移学习的关键。
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3),
)

# 第一阶段先冻结 ResNet50 主干网络。
# trainable=False 表示这些层的参数先不更新。
#
# 这样做的原因：
# - 避免一开始就把预训练好的特征“学坏”
# - 先让我们自己新加的分类层适应当前任务
base_model.trainable = False

# 这里使用 Sequential 把“预训练主干 + 新的分类头”连起来。
# 分类头的结构说明：
# 1. GlobalAveragePooling2D
#    把每个特征图压缩成一个数字，参数更少，比 Flatten 更适合迁移学习
# 2. Dense(256, relu)
#    学习更适合当前三分类任务的组合特征
# 3. Dropout(0.5)
#    随机丢弃一部分神经元，减少过拟合
# 4. Dense(num_classes, softmax)
#    输出 3 个类别的概率
model = models.Sequential(
    [
        layers.Input(shape=(img_height, img_width, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# --- 4. 训练参数设置 ---
# 这里保留了你原先 CNN 脚本中的“类别加权”思想。
# 如果担心模型漏判 sandstorm，可以适当提高沙尘暴类别的权重。
#
# 含义：
# class_weights[index] = 2.0
# 表示如果这一类样本分类错了，损失会更大，模型会更重视它。
class_weights = {i: 1.0 for i in range(num_classes)}
for label, index in class_indices.items():
    if "sandstorm" in label.lower():
        class_weights[index] = 2.0

# ModelCheckpoint：
# 每当验证集表现更好时，就自动保存当前最优模型。
# 这样即使后面训练过拟合，我们仍然能保留最好的版本。
checkpoint = callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

# EarlyStopping：
# 如果验证集损失连续几轮都没有变好，就提前停止训练。
# 这样可以节省时间，也能减少过拟合。
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
)

# ReduceLROnPlateau：
# 如果验证集损失停滞，就自动把学习率调小。
# 学习率变小后，模型通常能做更细致的调整。
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1,
)

# 第一阶段：只训练新加的分类头
# 由于 base_model.trainable = False，
# 这一阶段 ResNet50 主干不会更新，只训练后面新接上的 Dense 层。
#
# 这里学习率可以相对大一点，因为新加层还是随机初始化的。
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

initial_epochs = 10
history_stage1 = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr],
)

# 第二阶段：微调 ResNet50 的高层
# 第一阶段完成后，新分类头已经初步适应了当前任务。
# 接下来可以让 ResNet50 的“后面一部分层”也参与训练，
# 让它学到更适合天气图像的数据特征。
base_model.trainable = True

# 这里不是把整个 ResNet50 全部放开，而是只解冻最后大约 30 层。
# 前面的层往往学到的是比较通用的边缘、纹理、颜色等低层特征，
# 这些通常不需要大改；后面的高层特征更接近具体任务，更值得微调。
#
# 为什么不全解冻？
# - 小数据集上容易过拟合
# - 训练更慢
# - 也更容易把预训练好的权重“破坏掉”
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 微调阶段学习率必须更小。
# 因为这时我们是在“精修”已经训练好的网络，而不是从头大幅更新。
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_epochs = 10
history_stage2 = model.fit(
    train_generator,
    epochs=initial_epochs + fine_tune_epochs,
    initial_epoch=len(history_stage1.history["loss"]),
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr],
)

# 正式测试前，重新加载训练过程中保存的“验证集最优模型”
# 而不是直接使用最后一个 epoch 的结果。
# 这通常会让最终测试结果更可靠。
model = tf.keras.models.load_model(best_model_path)

# --- 5. 最终评估 ---
print("\n--- Generating final evaluation report ---")

# evaluate 会给出测试集上的总体 loss 和 accuracy
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

test_generator.reset()
predictions = model.predict(test_generator)

# predictions 的形状类似：
# [样本数, 类别数]
# 每一行都是该图片属于各类别的概率。
# np.argmax(predictions, axis=1) 的作用是取概率最大的类别下标。
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# classification_report 会输出每个类别的：
# - precision（精确率）
# - recall（召回率）
# - f1-score
#
# 这是比单纯 accuracy 更细的评估方式，特别适合多分类任务。
report_text = classification_report(y_true, y_pred, target_names=labels)
print("\nClassification report:")
print(report_text)

with open(classification_report_path, "w", encoding="utf-8-sig") as report_file:
    report_file.write("ResNet50 Classification Report\n")
    report_file.write("=" * 50 + "\n")
    report_file.write(report_text)

with open(evaluation_metrics_path, "w", encoding="utf-8-sig") as metrics_file:
    metrics_file.write("ResNet50 Evaluation Metrics\n")
    metrics_file.write("=" * 50 + "\n")
    metrics_file.write(f"Test Loss: {test_loss:.6f}\n")
    metrics_file.write(f"Test Accuracy: {test_accuracy:.6f}\n")
    metrics_file.write(f"Classes: {labels}\n")
    metrics_file.write(f"Class Indices: {class_indices}\n")

# confusion_matrix（混淆矩阵）可以直观看出：
# - 哪一类最容易识别
# - 哪两类最容易混淆
cm = confusion_matrix(y_true, y_pred)
save_confusion_matrix_csv(cm, labels, confusion_matrix_csv_path)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("ResNet50 Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(confusion_matrix_image_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 保存最终模型，后续可以在预测脚本中直接加载。
model.save(final_model_path)
print(f"Saved final model to {final_model_path}")

# --- 6. 绘制学习曲线 ---
# 因为训练分成两个阶段，所以这里把两个阶段的历史记录拼接起来，
# 方便画成一张完整曲线。
acc, val_acc, loss, val_loss = save_training_history(
    history_stage1,
    history_stage2,
    training_history_path,
)

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
# 左图：准确率曲线
# 如果训练准确率很高、验证准确率明显偏低，可能表示过拟合。
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 右图：损失曲线
# 一般来说，训练和验证损失越低越好；
# 但如果训练损失持续下降而验证损失上升，也说明可能过拟合。
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nAll ResNet50 result files have been saved to:")
print(result_dir)
