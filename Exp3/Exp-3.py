import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm
import os
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# ========================
# 1. 配置设备（GPU/CPU）
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ========================
# 2. 定义 CNN 模型
# ========================
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入: 1x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1)  # 1x1
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x


# ========================
# 3. 加载和预处理 MNIST 数据集
# ========================
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
    ])

    # 训练集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # 测试集
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 从训练集分出验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    return train_subset, val_subset, test_dataset


# ========================
# 4. 训练模型
# ========================
def train_model_pytorch():
    print("正在加载 MNIST 数据集...")
    train_subset, val_subset, test_dataset = load_mnist_data()

    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = DigitClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # 训练参数
    num_epochs = 15
    best_val_acc = 0.0

    print("开始训练模型...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()

            pbar.set_postfix({
                'loss': f"{train_loss / total:.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  训练损失: {train_loss / len(train_loader):.4f}, 训练准确率: {100 * correct / total:.2f}%")
        print(f"  验证损失: {val_loss_avg:.4f}, 验证准确率: {val_acc:.2f}%")

        # 学习率调度
        scheduler.step(val_loss_avg)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'digit_cnn_best.pth')
            print(f"保存新最佳模型 (验证准确率: {val_acc:.2f}%)")

    # 最终评估
    model.load_state_dict(torch.load('digit_cnn_best.pth'))
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"\n最终测试准确率: {test_acc:.2f}%")
    print("模型已保存为 digit_cnn_best.pth")

    return model


# ========================
# 5. 加载预训练模型
# ========================
def load_pretrained_model():
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load('digit_cnn_best.pth', map_location=device))
    model.eval()
    print("✓ 模型加载成功")
    return model


# ========================
# 6. 增强的图像预处理（专为学号识别优化）
# ========================
def preprocess_image(image_path):
    """增强的图像预处理，适应不同光照和背景"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 增强对比度（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 自适应阈值（对不均匀光照更鲁棒）
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 形态学操作：去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 闭运算：填充数字内部的小孔
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2)

    return cleaned, gray


# ========================
# 7. 智能数字分割（针对10位学号优化）
# ========================
def extract_digits_smart(binary_img, expected_count=10):
    """
    智能提取数字，处理粘连、分离等问题
    expected_count: 期望的数字数量（默认10位学号）
    """
    # 找出所有轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("⚠ 未检测到任何轮廓")
        return []

    # 获取图像尺寸
    img_h, img_w = binary_img.shape

    # 过滤并提取候选区域
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 计算面积和纵横比
        area = w * h
        aspect_ratio = h / w if w > 0 else 0

        # 动态过滤条件
        min_area = img_h * img_w * 0.0001  # 至少占0.01%
        max_area = img_h * img_w * 0.1  # 最多占10%

        # 数字通常是竖长的，纵横比在0.8-4之间
        if (area > min_area and area < max_area and
                0.5 < aspect_ratio < 4 and
                h > 10 and w > 5):
            candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio
            })

    print(f"找到 {len(candidates)} 个候选区域")

    if len(candidates) == 0:
        return []

    # 按x坐标排序（从左到右）
    candidates = sorted(candidates, key=lambda c: c['bbox'][0])

    # 智能合并：处理被分割的数字
    digits = merge_split_digits(candidates, binary_img)

    # 智能分离：处理粘连的数字
    if len(digits) < expected_count:
        digits = split_merged_digits(digits, binary_img, expected_count)

    print(f"处理后检测到 {len(digits)} 个数字")

    return digits


def merge_split_digits(candidates, binary_img):
    """合并被错误分割的数字（如"1"被分成两部分）"""
    if len(candidates) <= 1:
        return candidates

    merged = []
    i = 0

    while i < len(candidates):
        current = candidates[i]
        x1, y1, w1, h1 = current['bbox']

        # 检查是否需要与下一个候选合并
        should_merge = False
        if i + 1 < len(candidates):
            next_cand = candidates[i + 1]
            x2, y2, w2, h2 = next_cand['bbox']

            # 合并条件：
            # 1. 水平距离很近（< 数字宽度的40%）
            # 2. 垂直位置相近（y坐标差异 < 数字高度的50%）
            # 3. 高度相近
            horizontal_gap = x2 - (x1 + w1)
            vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            height_ratio = max(h1, h2) / min(h1, h2) if min(h1, h2) > 0 else 999

            if (horizontal_gap < max(w1, w2) * 0.4 and
                    vertical_overlap > min(h1, h2) * 0.5 and
                    height_ratio < 1.5):
                should_merge = True

        if should_merge:
            # 合并两个区域
            next_cand = candidates[i + 1]
            x2, y2, w2, h2 = next_cand['bbox']

            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_w = max(x1 + w1, x2 + w2) - new_x
            new_h = max(y1 + h1, y2 + h2) - new_y

            merged.append({
                'bbox': (new_x, new_y, new_w, new_h),
                'area': new_w * new_h,
                'aspect_ratio': new_h / new_w if new_w > 0 else 0
            })

            i += 2  # 跳过已合并的下一个
        else:
            merged.append(current)
            i += 1

    return merged


def split_merged_digits(digits, binary_img, expected_count):
    """
    尝试分离粘连的数字
    如果检测到的数字少于预期，尝试将宽度较大的区域分割
    """
    if len(digits) >= expected_count:
        return digits

    result = []

    # 计算平均宽度
    avg_width = np.mean([d['bbox'][2] for d in digits])

    for digit in digits:
        x, y, w, h = digit['bbox']

        # 如果这个区域明显太宽（可能是多个数字粘连）
        if w > avg_width * 1.8:
            # 尝试等分
            num_splits = int(round(w / avg_width))
            split_width = w // num_splits

            for i in range(num_splits):
                split_x = x + i * split_width
                # 确保最后一个分割包含剩余部分
                split_w = w - i * split_width if i == num_splits - 1 else split_width

                result.append({
                    'bbox': (split_x, y, split_w, h),
                    'area': split_w * h,
                    'aspect_ratio': h / split_w if split_w > 0 else 0
                })
        else:
            result.append(digit)

    return result


def prepare_digit_for_recognition(binary_img, bbox):
    """将数字区域准备为模型输入格式"""
    x, y, w, h = bbox

    # 提取ROI
    digit_roi = binary_img[y:y + h, x:x + w]

    # 添加边距（让数字居中，类似MNIST）
    padding = 4
    padded = cv2.copyMakeBorder(
        digit_roi, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=0
    )

    # 调整到28x28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # 归一化
    normalized = resized.astype('float32') / 255.0

    # 转为张量 [1, 1, 28, 28]
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)

    # MNIST归一化
    tensor = (tensor - 0.1307) / 0.3081

    return resized, tensor.to(device)


# ========================
# 8. 识别数字（带置信度）
# ========================
def predict_digit_pytorch(model, digit_tensor):
    """使用PyTorch模型预测数字"""
    with torch.no_grad():
        output = model(digit_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()


# ========================
# 9. 主识别流程（优化版）
# ========================
def recognize_student_id(image_path, model, expected_digits=10, min_confidence=0.5):
    """
    识别学号的主流程

    参数:
        image_path: 图像路径
        model: 训练好的模型
        expected_digits: 期望的数字位数（默认10位）
        min_confidence: 最低置信度阈值
    """
    print(f"\n{'=' * 50}")
    print(f"开始识别学号: {image_path}")
    print(f"{'=' * 50}")

    # 1. 预处理图像
    try:
        binary_img, gray_img = preprocess_image(image_path)
        print("✓ 图像预处理完成")
    except Exception as e:
        print(f"✗ 图像处理失败: {e}")
        return None

    # 保存预处理图像用于调试
    cv2.imwrite("debug_preprocessed.png", binary_img)

    # 2. 提取数字
    digit_candidates = extract_digits_smart(binary_img, expected_digits)

    if len(digit_candidates) == 0:
        print("✗ 未检测到任何数字")
        print("提示: 请检查图像是否清晰，数字是否完整")
        return None

    # 3. 识别每个数字
    results = []
    for i, candidate in enumerate(digit_candidates):
        # 准备输入
        digit_img, digit_tensor = prepare_digit_for_recognition(
            binary_img, candidate['bbox']
        )

        # 预测
        pred, conf, probs = predict_digit_pytorch(model, digit_tensor)

        results.append({
            'index': i,
            'digit': pred,
            'confidence': conf,
            'probabilities': probs,
            'bbox': candidate['bbox'],
            'image': digit_img
        })

        # 输出识别结果
        status = "✓" if conf >= min_confidence else "⚠"
        print(f"{status} 数字 {i + 1}: {pred} (置信度: {conf:.3f})")

        if conf < min_confidence:
            # 显示前3个可能的结果
            top3_idx = np.argsort(probs)[-3:][::-1]
            print(f"   备选: {', '.join([f'{idx}({probs[idx]:.2f})' for idx in top3_idx])}")

    # 4. 验证数字数量
    if len(results) != expected_digits:
        print(f"\n⚠ 警告: 检测到 {len(results)} 个数字，期望 {expected_digits} 个")
        print("   可能原因: 图像质量问题、数字粘连或遮挡")

    # 5. 组合学号
    student_id = ''.join([str(r['digit']) for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])

    print(f"\n{'=' * 50}")
    print(f"识别结果: {student_id}")
    print(f"平均置信度: {avg_confidence:.3f}")
    print(f"数字数量: {len(results)}/{expected_digits}")
    print(f"{'=' * 50}\n")

    return {
        'student_id': student_id,
        'digits': results,
        'avg_confidence': avg_confidence,
        'binary_image': binary_img,
        'gray_image': gray_img
    }


# ========================
# 10. 可视化结果（针对10位数字优化）
# ========================
def visualize_results(image_path, recognition_result):
    """可视化识别结果"""
    if recognition_result is None:
        print("无识别结果可显示")
        return

    student_id = recognition_result['student_id']
    digits = recognition_result['digits']
    binary_img = recognition_result['binary_image']

    # 读取原图
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # 在原图上标注检测框
    annotated = original.copy()
    for i, digit_info in enumerate(digits):
        x, y, w, h = digit_info['bbox']
        color = (0, 255, 0) if digit_info['confidence'] >= 0.5 else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # 标注数字和置信度
        label = f"{digit_info['digit']}:{digit_info['confidence']:.2f}"
        cv2.putText(annotated, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # 创建图形（调整布局以适应10位数字）
    num_digits = len(digits)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, max(num_digits, 5), hspace=0.3, wspace=0.3)

    # 第一行：原图和标注图
    ax1 = fig.add_subplot(gs[0, :num_digits // 2])
    ax1.imshow(original_rgb)
    ax1.set_title("原始图像", fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, num_digits // 2:])
    ax2.imshow(annotated_rgb)
    ax2.set_title(f"检测结果 - 学号: {student_id}", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 第二行：预处理图像
    ax3 = fig.add_subplot(gs[1, :])
    ax3.imshow(binary_img, cmap='gray')
    ax3.set_title("预处理后的二值图像", fontsize=12)
    ax3.axis('off')

    # 第三行和第四行：每个数字的详细信息
    for i, digit_info in enumerate(digits):
        if i >= num_digits:
            break

        # 数字图像
        ax_digit = fig.add_subplot(gs[2, i])
        ax_digit.imshow(digit_info['image'], cmap='gray')

        conf_color = 'green' if digit_info['confidence'] >= 0.7 else 'orange' if digit_info[
                                                                                     'confidence'] >= 0.5 else 'red'
        ax_digit.set_title(
            f"#{i + 1}: {digit_info['digit']}\n置信度: {digit_info['confidence']:.3f}",
            fontsize=10, color=conf_color, fontweight='bold'
        )
        ax_digit.axis('off')

        # 概率分布
        ax_prob = fig.add_subplot(gs[3, i])
        probs = digit_info['probabilities']
        bars = ax_prob.bar(range(10), probs, color='skyblue', edgecolor='navy')

        # 高亮预测的数字
        bars[digit_info['digit']].set_color('red')

        ax_prob.set_ylim(0, 1)
        ax_prob.set_xticks(range(10))
        ax_prob.set_ylabel('概率', fontsize=8)
        ax_prob.set_xlabel('数字', fontsize=8)
        ax_prob.tick_params(labelsize=8)
        ax_prob.grid(axis='y', alpha=0.3)

    plt.savefig("recognition_result.png", dpi=150, bbox_inches='tight')
    print("✓ 可视化结果已保存至: recognition_result.png")

    return fig


# ========================
# 11. 主程序
# ========================
def main():
    print("\n" + "=" * 60)
    print("  10位学号识别系统 - PyTorch CNN")
    print("=" * 60 + "\n")

    # 1. 检查并加载模型
    model_path = 'digit_cnn_best.pth'
    if not os.path.exists(model_path):
        print("⚠ 未找到模型文件，开始训练...")
        model = train_model_pytorch()
    else:
        print("正在加载模型...")
        model = load_pretrained_model()

    # 2. 指定学号图像路径
    input_image_path = "student_id.jpg"

    # 检查文件是否存在
    if not os.path.exists(input_image_path):
        print(f"\n✗ 错误: 未找到图像文件 '{input_image_path}'")
        print("\n当前目录中的图像文件:")
        image_files = [f for f in os.listdir('.')
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if image_files:
            for i, file in enumerate(image_files, 1):
                print(f"  {i}. {file}")

            # 可选：让用户选择文件
            try:
                choice = input("\n请输入文件编号或完整文件名: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(image_files):
                    input_image_path = image_files[int(choice) - 1]
                else:
                    input_image_path = choice
            except:
                print("使用默认路径")
        else:
            print("  未找到任何图像文件")
            return

    # 3. 识别学号
    result = recognize_student_id(
        input_image_path,
        model,
        expected_digits=10,
        min_confidence=0.5
    )

    if result is None:
        print("\n 识别失败，请检查图像质量")
        return

    # 4. 可视化结果
    visualize_results(input_image_path, result)

    # 5. 保存结果到文件
    with open("recognition_result.txt", "w", encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("学号识别结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"识别的学号: {result['student_id']}\n")
        f.write(f"数字数量: {len(result['digits'])} 位\n")
        f.write(f"平均置信度: {result['avg_confidence']:.4f}\n\n")

        f.write("详细信息:\n")
        f.write("-" * 50 + "\n")
        for digit_info in result['digits']:
            f.write(f"位置 {digit_info['index'] + 1}: ")
            f.write(f"数字 {digit_info['digit']}, ")
            f.write(f"置信度 {digit_info['confidence']:.4f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("使用模型: PyTorch CNN\n")
        f.write(f"处理时间: {np.datetime64('now')}\n")

    print("✓ 详细结果已保存至: recognition_result.txt")

    # 6. 显示图形（可选）
    try:
        plt.show()
    except:
        print("提示: 无法显示图形界面，但结果已保存至文件")


# 运行主程序
if __name__ == "__main__":
    main()