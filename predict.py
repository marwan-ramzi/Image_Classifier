#     Basic usage: python predict.py /path/to/image checkpoint
#     Options:
#         Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
#         Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#         Use GPU for inference: python predict.py input checkpoint --gpu

# sample bash cmd: python predict.py './flowers/test/42/image_05696.jpg' '../saved_models/checkpoint.pth' 'cat_to_name.json'

import argparse
import torch
import torchvision
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.alexnet import AlexNet_Weights
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import json

# import argparse
# import torch
# import torchvision
# from torch import nn, optim
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils import data
# from PIL import Image
# import numpy as np
# import os, random
# import matplotlib
# import matplotlib.pyplot as plt
# import json


def loading_the_checkpoint(path):
    checkpoint = torch.load(path)
    if checkpoint['structure'] == "vgg16":
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif checkpoint['structure'] == "resnet50":
        model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif checkpoint['structure'] == "alexnet":
        model = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)
    
    #Freezing the parameters of the pre-trained model as required
    for param in model.parameters():
        param.requires_grad = False

    input_size          = checkpoint['input_size']
    output_size         = checkpoint['output_size']
    structure           = checkpoint['structure']
    learning_rate       = checkpoint['learning_rate']
    epochs              = checkpoint['epochs']
    optimizer           = checkpoint['optimizer']
    model.classifier    = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx  = checkpoint['class_to_idx']
    
    return model
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    
    return image

def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()
    # image = image.unsqueeze(0)

    img = img.to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(img)
        
    probability = torch.exp(logps).data
        
    probability, top_classes = probability.topk(topk, dim=1)
    
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]

    return probability.tolist()[0], predicted_flowers_list

def print_predictions(args):

    model = loading_the_checkpoint(args.model_filepath)

    if args.gpu and torch.cuda.is_available():          #user choosed gpu and is available
        device = 'cuda'                                    
    if args.gpu and not(torch.cuda.is_available()):     #user choosed gpu and is not available
        device = 'cpu'                                  
        print("GPU is not available. CPU is used instead.")
    else:                                               #user did not choose gpu
        device = 'cpu'

    model = model.to(device)

    # print(model.class_to_index)

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # predict image
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
          print("#{: <3} {: <25} Prob: {:.2f}%".format(i, top_classes[i], top_ps[i]*100))
    
if __name__ == '__main__':
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='image_filepath', help="This is a image file that you want to classify")
    parser.add_argument(dest='model_filepath', help="This is file path of a checkpoint file, including the extension")

    # optional arguments
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', help="This is a file path to a json file that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="This is the number of most likely classes to return, default is 5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    print_predictions(args)