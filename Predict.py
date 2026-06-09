import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['mathtext.fontset'] = 'cm'

# --- 1. 加载已训练的模型 ---
model_path = 'optimized_sandstorm_model.h5'
model = tf.keras.models.load_model(model_path)
print("模型加载成功！")

# --- 2. 定义图像参数（必须与训练时一致） ---
img_height = 150
img_width = 150

# --- 3. 定义类别标签 ---
class_names = ['cloudy', 'sandstorm', 'sunny']  # 对应训练时的文件夹名称

# --- 4. 预处理单张图像函数 ---
def preprocess_image(image_path):
    """
    加载并预处理单张图像
    """
    # 加载图像
    img = load_img(image_path, target_size=(img_height, img_width))
    
    # 转换为数组
    img_array = img_to_array(img)
    
    # 归一化（除以255，范围0-1）
    img_array = img_array / 255.0
    
    # 添加批次维度（模型期望 (batch_size, height, width, channels) 的形状）
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- 5. 单张图像预测函数 ---
def predict_single_image(image_path, show_plot=True):
    """
    对单张图像进行预测并显示结果
    """
    # 预处理图像
    img_array = preprocess_image(image_path)
    
    # 进行预测
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # 显示结果（仅在需要时）
    if show_plot:
        # 显示原图
        img = Image.open(image_path)
        plt.figure(figsize=(10, 4))
        
        # 左图：原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'原始图像: {os.path.basename(image_path)}')
        plt.axis('off')
        
        # 右图：预测结果柱状图
        plt.subplot(1, 2, 2)
        colors = ['#FF6B6B', '#FFC93B', '#4ECDC4']
        bars = plt.bar(class_names, predictions[0], color=colors)
        plt.ylabel('置信度')
        plt.title(f'预测结果: {predicted_class}\n置信度: {confidence:.2f}%')
        plt.ylim([0, 1])
        
        # 在柱子上显示百分比
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{predictions[0][i]*100:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    # 打印详细结果
    print(f"\n{'='*50}")
    print(f"图像: {os.path.basename(image_path)}")
    print(f"预测类别: {predicted_class}")
    print(f"置信度: {confidence:.2f}%")
    print(f"\n各类别概率分布:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {predictions[0][i]*100:.2f}%")
    print(f"{'='*50}\n")
    
    return predicted_class, confidence, predictions[0]

# --- 6. 批量预测函数 ---
def predict_batch_from_folder(folder_path):
    """
    对指定文件夹中的所有图像进行预测
    """
    results = []
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"文件夹 {folder_path} 中没有找到图像文件！")
        return
    
    print(f"\n找到 {len(image_files)} 张图像，开始预测...\n")
    print("提示：批量预测时已禁用图形显示，结果将直接输出到终端。\n")
    
    # 逐个预测
    for image_path in image_files:
        try:
            # 批量预测时强制 show_plot=False
            predicted_class, confidence, all_probs = predict_single_image(image_path, show_plot=False)
            results.append({
                'filename': os.path.basename(image_path),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probs': all_probs
            })
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
    
    return results

# --- 7. 使用示例 ---
if __name__ == "__main__":
    print("沙尘暴天气识别系统 - 预测模块\n")
    
    # 方式一：预测单张图像
    print("请选择预测方式:")
    print("1. 预测单张图像")
    print("2. 预测整个文件夹")
    
    choice = input("\n请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        image_path = input("请输入图像文件的完整路径: ").strip()
        if os.path.exists(image_path):
            predict_single_image(image_path, show_plot=True)
        else:
            print(f"文件 {image_path} 不存在！")
    
    elif choice == '2':
        folder_path = input("请输入文件夹路径: ").strip()
        if os.path.exists(folder_path):
            results = predict_batch_from_folder(folder_path)
            
            # 统计准确率（如果文件夹结构是 weather_type/image.jpg）
            correct = 0
            for result in results:
                parent_folder = os.path.basename(os.path.dirname(image_path))
                if result['predicted_class'] == parent_folder:
                    correct += 1
            
            print(f"\n预测完成！")
            if len(results) > 0:
                print(f"总计: {len(results)} 张图像")
        else:
            print(f"文件夹 {folder_path} 不存在！")
    
    else:
        print("无效的选择！")
