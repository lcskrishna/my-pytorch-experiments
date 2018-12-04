import os
import sys
import argparse
import numpy as np

## Pytorch modules.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

fruit_names = [
    'Apple Braeburn',
    'Apple Golden 1',
    'Apple Golden 2',
    'Apple Golden 3',
    'Apple Granny Smith',
    'Apple Red 1',
    'Apple Red 2',
    'Apple Red 3',
    'Apple Red Delicious',
    'Apple Red Yellow',
    'Apricot',
    'Avocado',
    'Avocado ripe',
    'Banana',
    'Banana red',
    'Cactus fruit',
    'Carambula',
    'Cherry',
    'Clementine',
    'Cocos',
    'Dates',
    'Granadilla',
    'Grape Pink',
    'Grape White',
    'Grape White 2',
    'Grapefruit Pink',
    'Grapefruit White',
    'Guava',
    'Huckleberry',
    'Kaki',
    'Kiwi',
    'Kumquats',
    'Lemon',
    'Lemon Meyer',
    'Limes',
    'Litchi',
    'Mandarine',
    'Mango',
    'Maracuja',
    'Nectarine',
    'Orange',
    'Papaya',
    'Passion Fruit',
    'Peach',
    'Peach Flat',
    'Pear',
    'Pear Abate',
    'Pear Monster',
    'Pear Williams',
    'Pepino',
    'Pineapple',
    'Pitahaya Red',
    'Plum',
    'Pomegranate',
    'Quince',
    'Raspberry',
    'Salak',
    'Strawberry',
    'Tamarillo',
    'Tangelo'
]

class FruitNet(nn.Module):
    def __init__(self):
        super(FruitNet, self).__init__()
        self.conv1 = nn.Conv2d(3,32,5, padding=2, stride=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3, padding=2, stride=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 26 * 26, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 60)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

def run_training(train_data, train_labels):
    net = FruitNet()
    print (net)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    ## Train the network.
    for epoch in range(1):
        running_loss = 0.0
        print ("DEBUG: Length of the train lables is : {}".format(len(train_labels)))

        for i in range(len(train_labels)):
            #inputs , labels = np.array(train_data[i]), np.array(train_labels[i])
            inputs = np.array(train_data[i], dtype=np.float32)
            label = np.array([fruit_names.index(train_labels[i])], dtype=np.int32)

            inputs = np.expand_dims(inputs, axis=0)

#            print (inputs.shape)
#            print (label.shape)

            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(np.array(label))
            labels = torch.tensor(labels, dtype = torch.long)

#            print (inputs)
#            print (labels)
    
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print ("INFO: Iteration {} and loss : {}".format(i, running_loss))

    print ("INFO: finished training")
    
def main():
    data_path = args.data_path
    train_data_path = os.path.join(os.path.abspath(data_path), 'train_data.npy')
    train_labels_path = os.path.join(os.path.abspath(data_path), 'train_labels.npy')
    validation_data_path = os.path.join(os.path.abspath(data_path), 'validation_data.npy')
    validation_labels_path = os.path.join(os.path.abspath(data_path), 'validation_labels.npy')

    train_data = np.load(train_data_path)
    train_labels = np.load(train_labels_path)
    validation_data = np.load(validation_data_path)
    validation_labels = np.load(validation_labels_path)

    print ("INFO: Train data size is : {}".format(train_data.shape))
    print ("INFO: Train labels size is : {}".format(train_labels.shape))
    print ("INFO: validation data size is : {}".format(validation_data.shape))
    print ("INFO: validation labels size is : {}".format(validation_labels.shape))
    print ("INFO: Total number of classes : {}".format(len(fruit_names)))

    run_training(train_data, train_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Data path for npy files')
    
    args = parser.parse_args()
    main()
