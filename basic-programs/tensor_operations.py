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


## Operations...

y = torch.rand(5,3)
print (x + y)

print ("syntax 2 ")
print (torch.add(x,y))

result = torch.empty(5,3)
torch.add(x,y, out=result)
print (result)

print ("inplace addition")
y.add_(x)
print (y)

## can use standard numpy like bells accessing the tensor.
print (x)
print (x[:,1])

## resizing and reshaping tools.
print ("resize and reshape examples.")
x = torch.randn(4,4)
print (x)
y = x.view(16)
print (y)

z = x.view(-1,8)
print(z)
print (x.size(), y.size(), z.size())

## one element item.
x = torch.randn(1)
print (x)
print (x.item())

#### NUMPY BRIDGE
a = torch.ones(5)
print (a)

b = a.numpy()
print (b)

a.add_(1)
print(a)
print(b)

print ("Converting numpy array to tensor.")
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print (a)
print (b)


