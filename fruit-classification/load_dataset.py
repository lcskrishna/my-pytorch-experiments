import argparse
import os
import sys
import numpy as np
import cv2
import glob

print ("INFO: all the modules are imported.")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help='Path to the dataset folder')

args = parser.parse_args()

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

image_path = args.dataset

## Creation of training data.
train_data = []
train_labels = []
for fruit in fruit_names:
    print (fruit)
    folder_path = os.path.join(image_path, "Training", fruit)
    images = os.listdir(folder_path)

    for i in range(len(images)):
        final_path = os.path.join(folder_path, images[i])
        img =  cv2.imread(final_path, cv2.IMREAD_COLOR)
        dims = np.shape(img)
        img = np.reshape(img, (dims[2], dims[0], dims[1]))
        train_data.append(img)
        train_labels.append(fruit)

train_data = np.array(train_data)
print (train_data.shape)
train_labels = np.array(train_labels)
print (train_labels)

print ("OK: Training data created.")


## saving the data into a file.
np.save('train_data.npy', train_data)
check = np.load('train_data.npy')
np.save('train_labels.npy', train_labels)
check2 = np.load('train_labels.npy')

print (check.shape)
print (check2.shape)

validation_data = []
validation_labels = []
for fruit in fruit_names:
    print (fruit)
    folder_path = os.path.join(image_path, "Validation")
    images = os.listdir(folder_path)
    
    for i in range(len(images)):
        final_path = os.path.join(folder_path, images[i])
        img = cv2.imread(final_path, cv2.IMREAD_COLOR)
        dims = np.shape(img)
        img = np.reshape(img, (dims[2], dims[0], dims[1]))
        validation_data.append(img)
        validation_labels.append(fruit)

validation_data = np.array(validation_data)
print (validation_data.shape)
validation_labels = np.array(validation_labels)
print (validation_labels.shape)


## saving the data into a file.
np.save('validation_data.npy', validation_data)
check = np.load('validation_data.npy')
np.save('validation_labels.npy', validation_labels)
check2 = np.load('validation_labels.npy')

print (check.shape)
print (check2.shape)
