from __future__ import print_function
import torch

## construct a 5x3 matrix uninitalized.
x = torch.empty(5,3)
print (x)

##randomly initialized matrix.
print ("Randomly initiailzed matrix.")
x = torch.rand(5,3)
print(x)

## matrix filled with zeros and dtype long.
x = torch.zeros(5,3, dtype= torch.long)
print (x)

## create a tensor directly from data.
x = torch.tensor([5, 5.3])
print (x)

## create a tensor based on existing tensor.
x = x.new_ones(5,3, dtype = torch.double)
print (x)

x = torch.randn_like(x, dtype = torch.float)
print (x)
print (x.size())

