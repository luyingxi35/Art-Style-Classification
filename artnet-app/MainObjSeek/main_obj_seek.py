import os
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # 取消限制

def get_main_object_patch_yolo(image_path, patch_size=180, model=None):
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # 先尝试用 PIL 打开图片，若失败则跳过
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
    except Exception as e:
        print(f"跳过损坏图片: {image_path}, 错误: {e}")
        return None
    # 用 numpy 数组传递给 YOLO，避免 EXIF 问题
    import numpy as np
    img_np = np.array(img)
    results = model(img_np)
    boxes = results.xyxy[0]
    if len(boxes) == 0:
        return None
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    idx = areas.argmax()
    x1, y1, x2, y2 = map(int, boxes[idx][:4])
    patch = img.crop((x1, y1, x2, y2))
    patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
    return patch

def get_center_patch(image, patch_size):
    w, h = image.size
    left = (w - patch_size) // 2
    top = (h - patch_size) // 2
    right = left + patch_size
    bottom = top + patch_size
    return image.crop((left, top, right, bottom))

def process_folder_yolo(input_folder, output_folder, patch_size=180):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
           # 如果目标文件已存在则跳过
        if os.path.exists(out_path):
            continue
        patch = get_main_object_patch_yolo(in_path, patch_size, model)
        if patch is not None:
            patch.save(out_path)
            #print(f"{fname}: YOLO主体patch已保存")
        else:
            img = Image.open(in_path).convert('RGB')
            min_side = min(img.size)
            center_patch = get_center_patch(img, min_side)
            center_patch = center_patch.resize((patch_size, patch_size), Image.LANCZOS)
            center_patch.save(out_path)
            #print(f"{fname}: YOLO未检测到主体，保存中心patch")

def process_all_train_folders(base_folder='../raw_data', output_folder='object_patch', patch_size=180):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    for i in range(1, 10):
        input_folder = os.path.join(base_folder, f'train_{i}')
        if not os.path.exists(input_folder):
            continue
        for fname in os.listdir(input_folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            # 如果目标文件已存在则跳过
            if os.path.exists(out_path):
                continue
            patch = get_main_object_patch_yolo(in_path, patch_size, model)
            if patch is not None:
                patch.save(out_path)
            else:
                img = Image.open(in_path).convert('RGB')
                min_side = min(img.size)
                center_patch = get_center_patch(img, min_side)
                center_patch = center_patch.resize((patch_size, patch_size), Image.LANCZOS)
                center_patch.save(out_path)

# 用法
process_all_train_folders(base_folder='../raw_data', output_folder='object_patch', patch_size=180)

# 用法
#process_folder_yolo('../VGG16/trainie', 'object_patch', patch_size=180)

process_folder_yolo('../raw_data/test', 'object_patch', patch_size=180)