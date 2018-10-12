from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets , models, transforms
import time
import os
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type = str, required=True, help = 'Path to the data directory')

args = parser.parse_args()

print ("INFO: all imports done.")

## Load the data and apply transforms.
data_transforms = {
    'train' : transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

    'val' : transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

data_dir = args.data_dir

image_datasets = { x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)  for x in ['train', 'val']}
dataset_sizes = {x : len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print ("OK: selected device: {} ".format(device))

### Training the dataset.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print ('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print ('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            
            ## Iterate over the data.
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print ('{} Loss: {:4f} Accuracy: {:4f}'.format(phase, epoch_loss, epoch_acc))
        
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print ()

    elapsed_time = time.time() - since
    print ("INFO: Training complete in {:0f}m {:0f}s".format(elapsed_time//60, elapsed_time % 60))
    print ("Best validation accuracy: {:4f}".format(best_acc))


    ## Load best model weights.
    model.load_state_dict(best_model_wts)
    return model


### Evaluate the model.
total_corrects = 0
def evaluate_model(model):
    was_training = model.training
    model.eval()
 
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total_corrects += torch.sum(preds == labels.data)


#accuracy_validation = total_corrects / dataset_sizes['val']
#print ("OK: validation accuracy is : {}".format(accuracy_validation))

### Fine tuning the model.
model_ft = models.resnet18(pretrained = True)
num_ftrs = model_ft.fc.in_features
print ("DEBUG: num features are : {}".format(num_ftrs))

model_ft.fc = nn.Linear(num_ftrs, 2)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

### Train the model and evaluate.
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
#evaluate_model(model_ft)
#accuracy_validation = total_corrects / dataset_sizes['val']
#print ("OK: validation accuracy is : {}".format(accuracy_validation))
              
