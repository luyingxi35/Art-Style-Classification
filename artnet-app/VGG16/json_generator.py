import os
import json
import numpy as np
from PIL import Image, ImageFile
import tensorflow as tf
from tensorflow import keras
import shutil

class JSONGenerator:
    def __init__(self, model_path, img_size=180, classes=None, patch_dir="patches_tmp"):
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size
        self.patch_dir = patch_dir
        self.classes = classes or [
            'Minimalism', 'Romanticism', 'Rococo', 'Post_Impressionism', 'Art_Nouveau_Modern',
            'Renaissance', 'Pointillism', 'Realism', 'Ukiyo_e', 'Symbolism', 'Baroque', 'Cubism',
            'Abstract', 'Pop_Art', 'Impressionism', 'Expressionism', 'Color_Field_Painting'
        ]
        os.makedirs(self.patch_dir, exist_ok=True)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def preprocess_and_patch(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size * 2, self.img_size * 2))
        patches = []

        # 原图resize到img_size*img_size
        original = img.resize((self.img_size, self.img_size))
        patches.append(original)

        # 四角patch
        for (x, y) in [(0, 0), (self.img_size, 0), (0, self.img_size), (self.img_size, self.img_size)]:
            patch = img.crop((x, y, x + self.img_size, y + self.img_size))
            patches.append(patch)

        # 中心patch
        center_x = self.img_size // 2
        center_y = self.img_size // 2
        patch = img.crop((center_x, center_y, center_x + self.img_size, center_y + self.img_size))
        patches.append(patch)

        # 保存patches
        patch_paths = []
        for i, patch in enumerate(patches):
            patch_path = os.path.join(self.patch_dir, f"patch_{i}.jpg")
            patch.save(patch_path)
            patch_paths.append(patch_path)
        return patch_paths

    def predict_patches(self, patch_paths):
        data = []
        for patch_path in patch_paths:
            img = Image.open(patch_path).convert('RGB').resize((self.img_size, self.img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            data.append(arr)
        data = np.array(data)
        preds = self.model.predict(data)
        return preds

    def sparse_top5(self, pred):
        # 只保留前5个最大概率，其余置零
        top5_idx = np.argsort(pred)[-5:][::-1]
        sparse = np.zeros_like(pred)
        sparse[top5_idx] = pred[top5_idx]
        return sparse.tolist()
    
    @staticmethod
    def label_to_one_hot(label, classes):
        """将类别名或索引映射为独热向量"""
        if isinstance(label, str):
            idx = classes.index(label)
        else:
            idx = int(label)
        one_hot = [0] * len(classes)
        one_hot[idx] = 1
        return one_hot

    def generate_json(self, image_path, output_json, label=None):
        patch_paths = self.preprocess_and_patch(image_path)
        preds = self.predict_patches(patch_paths)
        sparse_preds = [self.sparse_top5(pred) for pred in preds]
        result = {
            "input": os.path.basename(image_path),
            "scores": sparse_preds,
            "label": self.label_to_one_hot(label, self.classes) if label is not None else None
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        #print(f"Saved prediction JSON to {output_json}")
        # 清空patches_tmp目录
        shutil.rmtree(self.patch_dir)
        os.makedirs(self.patch_dir, exist_ok=True)

# 用法示例
if __name__ == "__main__":
    generator = JSONGenerator(
        model_path="fine_tuned_VGG16_180x180.h5",
        img_size=180
    )
    # 输入图片路径和输出json路径
    generator.generate_json("trainie/182.jpg", "example.json", label="Minimalism")