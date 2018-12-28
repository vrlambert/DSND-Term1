# contains functions for working with the image classifier model
from torchvision import models
from torch import nn
import torch
from PIL import Image

from image_utils import process_image

# for easier use I split the load model function into two for the command line files
def build_model(arch, hidden, classes):
    # download the same base architecture
    if arch=='vgg13':
        model = models.vgg13(pretrained=True)
    elif arch=='vgg16':
        model = models.vgg16(pretrained=True)
    elif arch=='vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print('Model architecture not supported')
        raise 
       
    # freeze the params
    for param in model.parameters():
        param.requires_grad = False
    
    # build the classifier
    classifier = nn.Sequential(nn.Linear(25088, hidden),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(hidden, classes),
                           nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    return model

def load_model(path):
    # load the checkpoint
    checkpoint = torch.load(path)
    
    # build the model
    model = build_model(checkpoint['arch'], checkpoint['hidden'], len(checkpoint['class_to_idx']))
    
    # save the class to index map
    model.class_to_idx = checkpoint['class_to_idx']
    
    # load the weights
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def save_model(model, path, arch, hidden, class_to_idx):
    # saves the model to the specified path
    model.class_to_idx = class_to_idx
    model.eval()
    torch.save({'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'hidden':hidden,
                'arch':arch},
               path)
    
def validation(model, dataloader, criterion, device):
    # Function checks the loss and accuracy on the given dataset
    model.eval()
    
    valid_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            valid_loss += criterion(outputs, labels).item()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return valid_loss, correct/total

def train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs=5):
    # Print control
    print_every = 20
    steps = 0
    
    # set model to train
    model.train()
    
    # send model to device
    model.to(device)
    
    print('Training Started on {}'.format(device))
    
    # Loop for each epoch
    for e in range(epochs):
        # Hold loss for printing
        running_loss = 0
        model.train()
        
        # iterate over the training dataset
        for inputs, labels in train_loader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss, accuracy = validation(model, valid_loader, criterion, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}  ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}  ".format(valid_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy))

                running_loss = 0
                model.train()
                
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im = Image.open(image_path)
    tensor= torch.from_numpy(process_image(im)).type(torch.FloatTensor).unsqueeze_(0)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        probs=torch.exp(model.forward(tensor.to(device)))
        top_probs, top_labs = probs.topk(topk)
        
    return top_probs.tolist()[0], top_labs.tolist()[0]