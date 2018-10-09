import torch
import time

dtype = torch.float
device= torch.device("cpu")

N,D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, dtype=dtype, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=dtype, device=device, requires_grad=True)

learning_rate = 1e-06
start_time = time.time()

for i in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print (i, loss.item())

    ## compuation of backward using autograd package.
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

elapsed_time = time.time() - start_time
print ("INFO: total time using autograd computation is : {}".format(elapsed_time))

