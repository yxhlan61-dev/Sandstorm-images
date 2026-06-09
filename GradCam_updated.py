import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# ===== 1. 加载模型 =====
model = tf.keras.models.load_model('optimized_sandstorm_model.h5')
# 显式 build 以便访问输出张量
model.build((None, 150, 150, 3))

class_names = ['cloudy', 'sandstorm', 'sunny']
img_size = (150, 150)

# ===== 2. 预处理图片 =====
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ===== 3. 寻找最后一个卷积层 =====
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError('模型中没有找到 Conv2D 层')

last_conv_layer_name = find_last_conv_layer(model)
print('最后一个卷积层名称：', last_conv_layer_name)

# ===== 4. 生成 Grad-CAM 热力图 =====
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 手动定义从输入到目标的传播
    last_conv_layer = model.get_layer(last_conv_layer_name)
    layers = model.layers
    
    with tf.GradientTape() as tape:
        x = img_array
        conv_output = None
        for layer in layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
        
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), preds.numpy() 

# ===== 5. 叠加并显示 =====
def save_and_display_gradcam(img_path, heatmap, preds, class_names, alpha=0.4):
    img = image.load_img(img_path)
    img_to_arr = image.img_to_array(img)

    heatmap_rescaled = np.uint8(255 * heatmap)
    # 使用新版方法获取 colormap
    jet = plt.colormaps.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_rescaled]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_to_arr.shape[1], img_to_arr.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha * 255 + img_to_arr
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    superimposed_img = image.array_to_img(superimposed_img)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))
    
    # 1. 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_to_arr.astype('uint8'))
    plt.title('原图')
    plt.axis('off')

    # 2. 显示叠加结果
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('热力图叠加结果')
    plt.axis('off')

    # 在底部标注各类别概率
    probs = preds[0]
    prob_text = f"{class_names[0]}: {probs[0]:.2%} | {class_names[1]}: {probs[1]:.2%} | {class_names[2]}: {probs[2]:.2%}"
    plt.figtext(0.5, 0.05, f"各类别预测概率\n{prob_text}", ha="center", fontsize=12, bbox=dict(facecolor='orange', alpha=0.1, pad=5))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

# ===== 6. 执行 =====
# 请根据实际情况修改图片路径
img_path = r'data\\test\\cloudy\\c0993.jpg' 
img_array = get_img_array(img_path, img_size)

heatmap, preds = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
save_and_display_gradcam(img_path, heatmap, preds, class_names)
