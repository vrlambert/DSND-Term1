import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch

from model import build_model, save_model, validation, train_model


parser = argparse.ArgumentParser(description="Train a neural network")

parser.add_argument('data_dir', action='store')
parser.add_argument('--save_dir', action='store', default='checkpoint.pth')
parser.add_argument('--arch', action='store', default='vgg16')
parser.add_argument('--learning_rate', action='store',type=float, default=0.002)
parser.add_argument('--hidden_units', action='store', type=int, default=4096)
parser.add_argument('--epochs', action='store', type=int, default=7)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

device = torch.device('cuda' if args.gpu else 'cpu')

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

model = build_model(args.arch, args.hidden_units, 102)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

train_model(model, train_loader, valid_loader, optimizer, criterion, device, epochs=args.epochs)
print('Finished Training')

_, accuracy = validation(model, test_loader, criterion, device)
print('The test set accuracy is {:.4f}'.format(accuracy))

print('Saving model')
save_model(model, args.save_dir, args.arch, args.hidden_units, train_dataset.class_to_idx)
print('Model saved at {}'.format(args.save_dir))