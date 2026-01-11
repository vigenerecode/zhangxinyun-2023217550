import cv2
import numpy as np


# ====================== 核心处理模块 ======================
# 核心函数：颜色筛选
def color_filter(img):
    """适配该图的颜色筛选：精准匹配黄色虚线+白色双白线"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将BGR色彩空间转为HSV（更利于颜色分割）

    # 白色车道线处理（右侧双白线）
    lower_white = np.array([0, 0, 180])  # 白色下限：H=0(全色), S=0(无饱和度), V=180(避免过亮)
    upper_white = np.array([180, 30, 255])  # 白色上限：H=180(全色), S=30(低饱和度), V=255(最大亮度)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)  # 创建白色掩码（0=非白色, 255=白色）

    # 黄色车道线处理（中间虚线）
    lower_yellow = np.array([15, 80, 120])  # 黄色下限：H=15(黄), S=80(中饱和度), V=120(中亮度)
    upper_yellow = np.array([35, 255, 255])  # 黄色上限：H=35(黄), S=255(高饱和度), V=255(最大亮度)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # 创建黄色掩码

    # 合并掩码（同时保留黄色和白色车道线）
    mask = cv2.bitwise_or(white_mask, yellow_mask)  # 逻辑或：白色+黄色区域=255
    filtered_img = cv2.bitwise_and(img, img, mask=mask)  # 应用掩码：仅保留目标颜色
    return filtered_img


# 核心函数：定义感兴趣区域(ROI)（排除道路外干扰）
def region_of_interest(img):
    """适配该图的ROI：覆盖道路区域，排除两侧建筑/树木"""
    height, width = img.shape[:2]  # 获取图像高度和宽度

    # 定义道路区域多边形（"远窄近宽"视角）
    # 顶点坐标：[左下, 左上(顶部收窄), 右上, 右下]
    roi_vertices = np.array([
        [(0, height),  # 左下角（图像最左下点）
         (width * 0.3, height * 0.4),  # 左上（顶部高度=图像40%处）
         (width * 0.7, height * 0.4),  # 右上（顶部高度=图像40%处）
         (width, height)]  # 右下角（图像最右下点）
    ], dtype=np.int32)  # 强制转为整数坐标

    mask = np.zeros_like(img)  # 创建与原图同尺寸的黑色掩码
    cv2.fillPoly(mask, [roi_vertices], 255)  # 填充多边形（道路区域=255）
    masked_img = cv2.bitwise_and(img, mask)  # 应用掩码：仅保留道路区域
    return masked_img


# 核心函数：分离左右车道线（区分左侧边线+中间黄线 vs 右侧双白线）
def separate_left_right_lines(lines, img_shape):
    """适配该图的车道线分类：区分'左侧边线+中间黄线'（负斜率）与'右侧双白线'（正斜率）"""
    left_lines = []  # 存储左侧车道线（含左侧白线+中间黄线）
    right_lines = []  # 存储右侧车道线（右侧双白线）
    height, width = img_shape[:2]  # 获取图像尺寸

    if lines is None:
        return left_lines, right_lines  # 无检测到线条时直接返回空列表

    for line in lines:
        # 解析线段坐标(x1,y1,x2,y2)
        x1, y1, x2, y2 = line.reshape(4)
        # 避免除以零（垂直线）
        if x2 - x1 == 0:
            continue
        # 计算斜率（y变化量/x变化量）
        slope = (y2 - y1) / (x2 - x1)

        # 过滤近乎水平的干扰线（该图车道线斜率较陡）
        if abs(slope) < 0.6:
            continue

        # 分类：负斜率=左侧车道线（向左倾斜），正斜率=右侧车道线（向右倾斜）
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))
    return left_lines, right_lines


# 核心函数：霍夫变换直线检测（适配黄色虚线分段特征）
def hough_line_detection(img):
    """使用霍夫变换进行直线检测，参数调整以适应黄色虚线特征"""
    lines = cv2.HoughLinesP(
        img,
        rho=1,  # 间距精度（像素单位）
        theta=np.pi / 180,  # 角度精度（1度）
        threshold=25,  # 检测阈值（降低阈值以识别短虚线）
        minLineLength=30,  # 最小线段长度（30像素，适配短虚线）
        maxLineGap=40  # 最大间隙（40像素，允许拼接虚线分段）
    )
    return lines


# ====================== 辅助处理函数 ======================
# 转换为灰度图（边缘检测前必需）
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 降低计算复杂度


# 高斯模糊（减少噪声干扰）
def gaussian_blur(img, kernel_size=7):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)  # 7x7核，标准差=0


# Canny边缘检测（关键步骤：提取清晰边缘）
def canny_edge(img, low_threshold=30, high_threshold=100):
    return cv2.Canny(img, low_threshold, high_threshold)  # 低阈值30, 高阈值100


# 形态学增强（连接断开的边缘）
def morphological_enhance(img):
    kernel = np.ones((3, 3), np.uint8)  # 3x3卷积核
    return cv2.dilate(img, kernel, iterations=1)  # 膨胀操作（扩大边缘）


# 拟合车道线（用多项式拟合所有点）
def fit_lane_line(points, img_shape):
    """对检测到的车道线点进行直线拟合"""
    if len(points) == 0:
        return None  # 无有效点时返回None

    x, y = [], []  # 存储所有点的坐标
    for (x1, y1, x2, y2) in points:
        x.extend([x1, x2])  # 收集x坐标
        y.extend([y1, y2])  # 收集y坐标

    # 用y作为自变量，x作为因变量拟合直线（y在图像高度方向）
    fit = np.polyfit(y, x, 1)  # 一阶多项式拟合（直线）
    fit_fn = np.poly1d(fit)  # 生成多项式函数

    # 定义拟合直线的起点和终点（从道路顶部到图像底部）
    y_min = int(img_shape[0] * 0.4)  # 从图像高度40%处开始（避开顶部干扰）
    y_max = img_shape[0]  # 图像底部
    x_min = int(fit_fn(y_min))  # 计算起点x坐标
    x_max = int(fit_fn(y_max))  # 计算终点x坐标
    return (x_min, y_min, x_max, y_max)  # 返回拟合直线的起点和终点


# 绘制拟合后的车道线
def draw_fitted_lanes(img, left_line, right_line):
    """在原始图像上绘制拟合的车道线"""
    lane_img = np.zeros_like(img)  # 创建与原图同尺寸的黑色图像

    # 绘制左侧车道线（绿色，粗细12像素）
    if left_line is not None:
        cv2.line(lane_img, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), (0, 255, 0), 12)

    # 绘制右侧车道线（绿色，粗细12像素）
    if right_line is not None:
        cv2.line(lane_img, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), (0, 255, 0), 12)

    # 将车道线叠加到原图（0.8透明度）
    return cv2.addWeighted(img, 0.8, lane_img, 1, 0)


# ====================== 主流程 ======================
if __name__ == "__main__":
    img_path = "test.jpg"  # 测试图像路径（需替换为实际路径）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径：{img_path}")

    img_copy = np.copy(img)  # 避免修改原始图像
    img_shape = img_copy.shape  # 获取图像尺寸（高度, 宽度, 通道数）

    # ====================== 核心处理流程 ======================
    # 1. 颜色筛选：保留黄线+白线
    filtered_img = color_filter(img_copy)

    # 2. 转灰度图（边缘检测必需）
    gray_img = grayscale(filtered_img)

    # 3. 高斯模糊（减少噪声）
    blur_img = gaussian_blur(gray_img)

    # 4. Canny边缘检测（提取清晰边缘）
    edge_img = canny_edge(blur_img)

    # 5. 形态学增强（连接断开的边缘）
    enhance_img = morphological_enhance(edge_img)

    # 6. ROI裁剪（仅保留道路区域）
    roi_img = region_of_interest(enhance_img)

    # 7. 霍夫变换检测直线
    lines = hough_line_detection(roi_img)

    # 8. 分离左右车道线
    left_lines, right_lines = separate_left_right_lines(lines, img_shape)

    # 9. 拟合左右车道线
    left_lane = fit_lane_line(left_lines, img_shape)
    right_lane = fit_lane_line(right_lines, img_shape)

    # 10. 绘制最终车道线
    result_img = draw_fitted_lanes(img_copy, left_lane, right_lane)

    # ====================== 结果输出 ======================
    save_path = "lane_result_for_this_img.jpg"  # 保存路径
    cv2.imwrite(save_path, result_img)  # 保存结果图像
    print(f"结果图像已保存到项目目录：{save_path}")

    # 显示结果
    cv2.imshow("Lane Detection Result", result_img)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有窗口