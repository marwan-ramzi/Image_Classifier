#sample to run train.py: 
#python train.py './flower_data' --save_directory './saved_checkpoints' --model_arch 'alexnet'

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torchvision
from torchvision import transforms, datasets, models
import torch
from torch import nn, optim
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.alexnet import AlexNet_Weights
from PIL import Image


def data_transforms(args):
    
    train_dir = args.data_directory + "/train"
    valid_dir = args.data_directory + "/valid"
    # print(train_dir)

    train_transforms = transforms.Compose([transforms.RandomRotation(degrees = 25),
                                           transforms.RandomResizedCrop(size = (224,224)),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                std  = [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(size = 255),
                                           transforms.CenterCrop(size = 224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                                std  = [0.229, 0.224, 0.225])])


    train_data  = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data  = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)

    return trainloader, validloader, train_data.class_to_idx

def training(args, trainloader, validloader, class_to_idx):
        
    #Loading the pre-trained model   
    if args.model_arch == "vgg16":
        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        features = model.classifier[0].in_features
    elif args.model_arch == "alexnet":
        model = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)
        features = model.classifier[1].in_features

    #Freezing the parameters of the pre-trained model as required
    for param in model.parameters():
        param.requires_grad = False

    #getting the dimensions of features extracted by the convolutional layers of the model
    print(f"output of last layer in {args.model_arch}: {features}")  ##last layer output

    flower_categ = len(class_to_idx) #classifer output should be equal to the number of flower classes = 102

    #Defining the new classifier
    classifier = nn.Sequential(nn.Linear(in_features=features, out_features=args.hidden_units, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=args.dropout),
                            nn.Linear(in_features=args.hidden_units, out_features=flower_categ, bias=True),
                            nn.LogSoftmax(dim=1)
                            )

    model.classifier = classifier

    criterion = nn.NLLLoss() #defining the criterion as the Negative log likelihood 

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # running on gpu option
    if args.gpu and torch.cuda.is_available():          #user choosed gpu and is available
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):   #user choosed gpu and is not available
        device = 'cpu'                                  
        print("GPU is not available. CPU is used instead.")
    else:                                               #user did not choose gpu
        device = 'cpu'
    print(f"{device} is used to train model.")
            
    model.to(device)

    epochs = 3
    print_every = 20
    steps = 0

    for i in range(epochs):
        running_loss = 0
        steps = 0
        for inputs, labels in trainloader:
            steps += 1
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            
            #Sets the gradients of all optimized class torch.Tensors to zero
            optimizer.zero_grad()
                        
            #Forward the features to the output
            outputs = model.forward(inputs)

            #Calculating the loss and applying back propagation to modify weights and hyper parameters
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0 or steps == 1 or steps == len(trainloader):
                print(f"Epoch: {i+1}/{epochs} Batch % Complete: {(steps)*100/len(trainloader):.2f}%")

        #validation
        
        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model.forward(inputs)

                batch_loss = criterion(outputs, labels)
                valid_loss += batch_loss.item()
                
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                
                equals = top_class == labels.view(*top_class.shape)
                
                #equals.type(torch.FloatTensor):
                #Transforms "equals" the tensor of array of boolean to tensor of floating ones and zeros
                #Then get the means of these numbers and finaly transforms the mean from tensor to python number
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Epoch {i+1}/{epochs}.. "
                f"Train Loss: {running_loss/len(trainloader):.3f}.. "
                f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                f"Accuracy: {accuracy*100/len(validloader):.3f}%")
        running_loss = 0

    # Save the checkpoint 
    model.class_to_idx = class_to_idx
    checkpoint = {
                    'input_size': features,
                    'output_size': flower_categ,
                    'structure': args.model_arch,
                    'learning_rate': 0.001,
                    'classifier': model.classifier,
                    'epochs': args.epochs,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx
                }

    save_dir = args.save_directory + "/checkpoint.pth"
    torch.save(checkpoint, save_dir)
    print(f"model saved to {save_dir}")
    return True

if __name__ == '__main__':

    # Creating the parser
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(dest='data_dir', help="Directory of the train and valid images")
    
    # Optional arguments
    parser.add_argument('--save_dir', dest='save_dir', help="Directory of saved checkpoint.", default=r'C:\Users\Marwan\Desktop\aipnd-project-master')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--epochs', dest='epochs', default=3, type=int)
    parser.add_argument('--gpu', dest='gpu', help="This argument allows the user to train the model on the GPU via CUDA", action='store_true')
    parser.add_argument('--model', dest='model', help="This argument allows the user to choose from 2 different pretrained models.", default="vgg16", type=str, choices=['vgg16', 'alexnet'])
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=2048)
    parser.add_argument('--dropout', action="store", type=float, default=0.25)

    args = parser.parse_args()

    trainloader, validloader, class_to_idx = data_transforms(args)

    training(args, trainloader, validloader, class_to_idx)


    
    