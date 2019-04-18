from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import argparse
import string
import unicodedata
import torch


print ("INFO: All imports are complete.")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset directory")
args = parser.parse_args()

dataset_dir = args.dataset
print("INFO: Dataset is present in {}".format(dataset_dir))

def findFiles(path):
    return glob.glob(path)

files = findFiles(dataset_dir + "/names/*.txt")

print ("INFO: The below are the files present : ")
for i in range(len(files)):
    print (files[i])

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

print ("INFO: Total number of letters is : {}".format(n_letters))

## Convert unicode to ascii code.
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)

print (unicodeToAscii('Ślusàrski'))

## Build category lines dictionary.
category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles(dataset_dir + "/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print (category_lines['Italian'][:5])

### Converting the names to Tensors.
def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print (letterToTensor('J'))
print (lineToTensor('Jones').size())

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_layer, hidden_layer):
        combined = torch.cat((input_layer, hidden_layer), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True
    print ("OK: Can run the network on GPU")

if is_cuda:
    rnn = rnn.cuda()

## Testing program.
input_layer = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)

if is_cuda:
    input_layer = input_layer.cuda()
    hidden = hidden.cuda()
output, next_hidden = rnn(input_layer, hidden)

## Efficient program
input_layer = lineToTensor("Albert")
hidden = torch.zeros(1, n_hidden)
if is_cuda:
    hidden = hidden.cuda()
output, next_hidden = rnn(input_layer[0].cuda(), hidden)
print (output)
print (output.size())


#### Helper functions.
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    #print (top_n)
    #print (top_i)
    #print (top_i.size())
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print (categoryFromOutput(output))

## Quick way to get a training example.
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)],dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print ("Category = {} / Line = {}".format(category, line))

criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    if is_cuda:
        hidden = hidden.cuda()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    loss.backward()
    
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

## Run actual training.
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    if is_cuda:
        category_tensor = category_tensor.cuda()
        line_tensor = line_tensor.cuda()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'Yes' if guess == category else 'No (%s)' % category
        print ("%d %d%% (%s) %0.4f %s / %s %s" % (iter, iter/n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss/ plot_every)
        current_loss = 0

def evaluate(line_tensor):
    hidden = rnn.initHidden().cuda()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].cuda(), hidden)

    return output

def predict(input_line, n_predictions=3):
    print ("\n> %s" % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print ("(%0.2f) %s" % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Dovesky')
predict('Jackson')
predict('Satoshi')
