import numpy as np
from collections import OrderedDict
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def process_image(pil_image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''                       
    
    # pil_image = Image.open(image_path)
    
    width, height = pil_image.size
    ratio = float(width/height);
        
    if height > width:
        height = int(height * 256 / width)
        width = int(256)
    else:
        width = int(width * 256 / height)
        height = int(256)
        
    resized_image = pil_image.resize((width, height), Image.ANTIALIAS)
    
    # Crop center portion of the image
    x0 = (width - 224) / 2
    y0 = (height - 224) / 2
    x1 = x0 + 224
    y1 = y0 + 224
    crop_image = resized_image.crop((x0,y0,x1, y1))
    
    # Normalize:
    np_image = np.array(crop_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    return np_image.transpose(2,0,1)

def build_model_vgg(class_to_idx = {'Art Nouveau (Modern)': 0,
                                     'Baroque': 1,
                                     'Expressionism': 2,
                                     'Impressionism': 3,
                                     'Post-Impressionism': 4,
                                     'Rococo': 5,
                                     'Romanticism': 6,
                                     'Surrealism': 7,
                                     'Symbolism': 8},
                   hidden_units=12595,
                   input_size=25088,
                   output_size=9,
                   dropout = 0.5):
    # loading vgg network
    model = models.vgg19(pretrained=True) 
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    #classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    
    return model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=device)
    
    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden']
    class_to_idx = checkpoint['class_to_idx']
    
    model = build_model_vgg(class_to_idx,
                   hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, learning_rate, hidden_units, class_to_idx

def predict(image, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Calculate the class probabilities (softmax) for img
    model.eval()

    img = process_image(image)
    tensor_in = torch.from_numpy(img)

    tensor_in = tensor_in.float() 
    tensor_in = tensor_in.unsqueeze(0)

    model.to(device)
    inputs = tensor_in.to(device)
    
    with torch.no_grad():
        output = model.forward(inputs)
        
    output = torch.exp(output)
        
    topk_prob, topk_index = torch.topk(output, topk) 
    topk_prob = topk_prob.tolist()[0]
    topk_index = topk_index.tolist()[0]
    
    idx_to_cat = {value: key for key, value in model.class_to_idx.items()}
    
    top_cat = [idx_to_cat[ele] for ele in topk_index]

    return topk_prob, top_cat