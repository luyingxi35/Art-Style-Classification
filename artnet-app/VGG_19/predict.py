# load module
import json
import predict_utils
from predict_utils import process_image, load_checkpoint, predict
import argparse
import torch
import cv2
from PIL import Image
from PIL import ImageFile

parser = argparse.ArgumentParser(
    description='Parameters for predict')
parser.add_argument('--input', action="store",
                    dest="input", default = 'checkpoint5.pth')
parser.add_argument('--top_k', action="store",
                    dest="top_k", default = '8')
parser.add_argument('--image', action="store",
                    dest="image", default = '/root/autodl-tmp/.autodl/artnet-app/testie/13.jpg')

args = vars(parser.parse_args())

#imputs
image_path = args['image']
checkpoint = args['input']
topk = int(args['top_k'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class and indixes
cat_to_name = {0: 'Art Nouveau (Modern)',
                 1: 'Baroque',
                 2: 'Expressionism',
                 3: 'Impressionism',
                 4: 'Post-Impressionism',
                 5: 'Rococo',
                 6: 'Romanticism',
                 7: 'Surrealism',
                 8: 'Symbolism'}

# load the model
model, learning_rate, hidden_units, class_to_idx = load_checkpoint(checkpoint)

# prediction
# image = cv2.imread(image_path)
# if image is None:
#     raise ValueError(f"无法读取图像: {image_path}")
pil_image = Image.open(image_path)

probs, top_labels = predict(pil_image, model, 9)

# print results

res = "\n".join("{} {}".format(x, y) for x, y in zip(probs, top_labels))

print(res)