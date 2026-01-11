import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 1. 读取图像
def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # 转为 RGB 格式
    return np.array(img)  # 转为 numpy 数组 (H, W, 3)

# 2. 手动实现二维卷积（无内置函数）
def convolve_2d(image, kernel):
    H, W, C = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # 填充图像
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)

    # 输出尺寸
    out_H = H
    out_W = W
    output = np.zeros((out_H, out_W, C))

    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                region = padded[i:i + kh, j:j + kw, c]
                output[i, j, c] = np.sum(region * kernel)

    return output


# 3. Sobel 算子
def sobel_filter(image):
    # Sobel X: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # Sobel Y: [1, 2, 1; 0, 0, 0; -1, -2, -1]
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    # 分别对每个通道进行卷积
    grad_x = convolve_2d(image, sobel_x)
    grad_y = convolve_2d(image, sobel_y)

    # 合成梯度幅值
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 归一化到 [0, 255]
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    # 将灰度图扩展为三通道以便保存
    return np.stack([gradient_magnitude] * 3, axis=-1)


# 4. 给定卷积核滤波
def custom_kernel_filter(image):
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    return convolve_2d(image, kernel)


# 5. 颜色直方图计算与可视化（手动实现）
def compute_histogram(image):
    H, W, C = image.shape
    bins = 256
    hist = np.zeros((C, bins))

    for c in range(C):
        for i in range(H):
            for j in range(W):
                pixel_val = int(image[i, j, c])
                hist[c, pixel_val] += 1

    return hist

# 绘制并保存一幅颜色直方图的图像
def plot_histogram(hist, save_path):
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(hist[i], color=colors[i], label=f'{colors[i]}')
    plt.title("Color Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# 6. 提取纹理特征
def extract_texture_features(image_gray):
    # 转为灰度图（平均值）
    gray = image_gray.astype(np.float32) / 255.0
    gray = (gray * 255).astype(np.uint8)

    # 简化版本：计算局部均值和方差作为纹理特征
    H, W = gray.shape
    features = []

    # 滑动窗口计算局部统计量
    window_size = 5
    half = window_size // 2

    for i in range(half, H - half):
        for j in range(half, W - half):
            patch = gray[i - half:i + half + 1, j - half:j + half + 1]
            mean_val = np.mean(patch)
            var_val = np.var(patch)
            features.append([mean_val, var_val])

    # 返回所有局部特征的均值和标准差
    features = np.array(features)
    texture_mean = np.mean(features, axis=0)
    texture_std = np.std(features, axis=0)

    # 也可以保存整个特征矩阵
    return texture_mean, texture_std, features


# 主程序
def main():
    input_image_path = "your_photo.jpg"
    if not os.path.exists(input_image_path):
        print(f"Error: {input_image_path} not found!")
        return

    # 1. 加载图像
    image = load_image(input_image_path)
    print("Image loaded successfully.")

    # 2. Sobel 滤波
    sobel_result = sobel_filter(image)

    # 3. 自定义卷积核滤波
    custom_result = custom_kernel_filter(image)

    # 4. 颜色直方图
    hist = compute_histogram(image)
    plot_histogram(hist, "color_histogram.png")

    # 5. 纹理特征提取（使用灰度图）
    # 转为灰度图
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    texture_mean, texture_std, full_features = extract_texture_features(gray_image)

    # 保存纹理特征为 .npy 文件
    np.save("texture_features.npy", full_features)
    print("Texture features saved to texture_features.npy")

    # 6. 保存结果图像
    plt.imsave("sobel_filtered.png", sobel_result)
    plt.imsave("custom_filtered.png", custom_result.astype(np.uint8))
    plt.imsave("original_image.png", image)

    print("All outputs saved successfully.")
    print("Files generated:")
    print("  - sobel_filtered.png")
    print("  - custom_filtered.png")
    print("  - color_histogram.png")
    print("  - texture_features.npy")


# 运行主程序
if __name__ == "__main__":
    main()