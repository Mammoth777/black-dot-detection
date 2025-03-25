import cv2  # 导入OpenCV库，用于图像处理
import os  # 导入os库，用于文件和目录操作
import glob  # 导入glob库，用于文件路径模式匹配

# 参数设置 - 放在顶部便于调整
THRESHOLD_VALUE = 80  # 二值化阈值，越低检测越黑的部分
MIN_AREA = 20  # 最小黑点面积
MAX_AREA = 10000  # 最大黑点面积
CROP_HEIGHT = 640  # 裁剪高度

# 创建输出目录
output_dir = "cropped_spots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("temp", exist_ok=True)  # 确保temp目录存在

def process_image(image_path):
    """处理单个图像文件，检测黑点并裁剪"""
    # 获取文件名（不含扩展名）用于输出命名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 图像预处理 - 高斯模糊去噪
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 二值化处理 - 使用更低的阈值检测更黑的部分
    _, threshold = cv2.threshold(blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    
    # 查找连通区域（黑斑）
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 标记黑斑，添加面积过滤
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    original_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    spot_count = 0  # 该图像中检测到的黑点计数
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 面积过滤 - 使用全局定义的参数
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 标记黑点位置
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(output, (center_x, center_y), 2, (0, 255, 0), -1)
            
            # 计算裁剪区域（固定高度，使用全局参数）
            new_w = img.shape[1]
            new_h = CROP_HEIGHT
            new_x = 0
            
            # 计算使黑点垂直居中的y坐标
            center_y = y + h // 2
            new_y = max(0, center_y - new_h // 2)
            
            # 确保裁剪区域不超出图像底部边界
            if new_y + new_h > img.shape[0]:
                new_y = max(0, img.shape[0] - new_h)
            
            # 裁剪区域
            cropped = original_color[new_y:new_y+new_h, new_x:new_x+new_w]
            
            # 使用"原图名称-spot-n"格式保存裁剪区域
            output_filename = f"{output_dir}/{base_name}-spot-{spot_count}.png"
            cv2.imwrite(output_filename, cropped)
            spot_count += 1
    
    # 保存中间结果
    cv2.imwrite(f"temp/{base_name}-output.png", output)
    cv2.imwrite(f"temp/{base_name}-threshold.png", threshold)
    
    return spot_count

# 主程序
def main():
    # 获取inputs目录中的所有图像文件
    input_files = glob.glob("inputs/*.jpg") + glob.glob("inputs/*.png") + glob.glob("inputs/*.jpeg") + glob.glob("inputs/*.bmp")
    
    if not input_files:
        print("未在inputs目录找到图片文件")
        return
        
    total_files = len(input_files)
    total_spots = 0
    
    print(f"开始处理，共发现 {total_files} 张图片...")
    
    # 遍历处理每张图片
    for idx, image_path in enumerate(input_files, 1):
        # 简单打印当前处理的文件名和进度
        print(f"正在处理 {idx}/{total_files}: {os.path.basename(image_path)}")
        
        # 处理图片
        spots = process_image(image_path)
        total_spots += spots
        
        # 显示处理结果
        print(f"  检测到 {spots} 个黑点")
    
    # 打印处理总结
    print(f"\n处理完成！总共处理了 {total_files} 张图片，检测到 {total_spots} 个黑点")
    print(f"裁剪结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()
