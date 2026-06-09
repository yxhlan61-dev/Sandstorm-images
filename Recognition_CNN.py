import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['mathtext.fontset'] = 'cm'

# --- 1. 设置文件夹路径 ---
base_dir = r'd:\\桌面\\science research\\Sandstorm Picture\\data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# 设置图像的标准大小和每次处理的数量
img_height = 150 
img_width = 150  
batch_size = 32  

# --- 2. 图像预处理与增强 ---
# 既然有了3000张图片，我们可以加入稍微强一点的数据增强，让模型见多识广
train_datagen = ImageDataGenerator(
    rescale=1./255,          # 归一化像素值到 [0, 1]
    rotation_range=30,       # 旋转角度加大
    width_shift_range=0.2,   # 水平/垂直平移20%
    height_shift_range=0.2,  
    shear_range=0.2,         # 剪切变换20%
    zoom_range=0.3,          # 缩放范围30%
    horizontal_flip=True,    # 随机水平翻转
    fill_mode='nearest'      # 填充新创建像素的方法（最近邻填充）
)

# 验证集和测试集只需要归一化（注意：不要在这些集上做随机变换）
val_test_datagen = ImageDataGenerator(rescale=1./255)

# 加载数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# shuffle=False 很重要，否则后面预测结果和真实标签会对应不上
val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # 验证集可以打乱，测试集不打乱
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False 
)

# 获取类别索引字典
class_indices = train_generator.class_indices
labels = list(class_indices.keys())
num_classes = len(labels) # 自动检测类别数量

# --- 3. 搭建更深层的神经网络 ---
# 增加了 BatchNormalization 以缓解梯度泄露问题
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)), # 使用 Input 对象作为首层
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(), 
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(256, (3, 3), activation='relu'), 
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.6), 
    layers.Dense(num_classes, activation='softmax') # 动态设置输出层数量
])

# --- 4. 配置训练参数（针对“不漏掉沙尘暴”的需求） ---

# 赋权逻辑：给 sandstorm 及其增强文件夹更大的惩罚权重
class_weights = {i: 1.0 for i in range(num_classes)}
for label, index in class_indices.items():
    if 'sandstorm' in label:
        class_weights[index] = 2  # 所有包含 sandstorm 字样的文件夹都加权

# 配置模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'] # 移除 Recall，以最基本的监控开始
)

# --- 5. 训练模型（加入自动停止机制） ---
# 如果连续 3 次模型没有变好，则自动停止训练，保留最好的结果
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

epochs = 30
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    class_weight=class_weights,  # 应用类别权重
    callbacks=[early_stop]
)

# --- 6. 深度评估：准确率、召回率、混淆矩阵 ---
print("\n--- 正在生成最终评估报告 ---")

# 重置生成器，开始预测测试集
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1) # 预测的类别
y_true = test_generator.classes          # 真实的类别

# 1. 打印分类报告（包含 Precision, Recall, F1-score）
print("\n每个类别的性能评估如下：")
print(classification_report(y_true, y_pred, target_names=labels))

# 2. 绘制混淆矩阵（更直观地反应用图认成了谁）
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
plt.title('模型混淆矩阵 (Confusion Matrix)')
plt.ylabel('实际类别')
plt.xlabel('预测类别')
plt.show()

# 7. 保存最终优化版模型
model.save('optimized_sandstorm_model.h5')
print("优化版模型已保存：optimized_sandstorm_model.h5")

# --- 7. 画出学习曲线 ---
# 如果准确率一直上升，损失值一直下降，说明模型在变好
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='训练准确率')
plt.plot(epochs_range, val_acc, label='验证准确率')
plt.title('准确率变化图')
plt.xlabel('轮次 (Epoch)')
plt.ylabel('准确率')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='训练误差')
plt.plot(epochs_range, val_loss, label='验证误差')
plt.title('损失值变化图')
plt.xlabel('轮次 (Epoch)')
plt.ylabel('损失值')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() # 弹出一个画框展示结果