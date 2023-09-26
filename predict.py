#sample to run predict.py: 
#python predict.py 'flower_data/test/1/image_06743.jpg' './saved_checkpoints/vgg16' --cat_to_names 'cat_to_name.json' --topk 5 

import argparse
import torch
import torchvision
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.alexnet import AlexNet_Weights
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import json


def loading_the_checkpoint(path):
    checkpoint = torch.load(path + "/checkpoint.pth")
    if checkpoint['structure'] == "vgg16":
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
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
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    img = img.to(device)

    with torch.no_grad():
        output = model.forward(img)
        
    probability = torch.exp(output).data

    probs, classes = probability.topk(topk)

    probs = probs[0].tolist()
    flowers_labels = [cat_to_name[str(i+1)] for i in classes.tolist()[0]]
    
    return probs, flowers_labels

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

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # predict image
    probs, classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
          print(f"#{i: <3} {classes[i]: <25} Prob: {probs[i]*100:.2f}%")
    
if __name__ == '__main__':
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='image_filepath', help="Image file path required to classify")
    parser.add_argument(dest='model_filepath', help="Model file path of a saved checkpoint file")

    # optional arguments
    parser.add_argument('--cat_to_names', dest='cat_to_names', help="JSON file path that maps flower labels", default='cat_to_name.json')
    parser.add_argument('--topk', dest='topk', help="This argument allows the user to know the top k flower categories classification, default is k=5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="This argument allows the user to train the model on the GPU via CUDA", action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    print_predictions(args)

    