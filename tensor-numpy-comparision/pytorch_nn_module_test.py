import torch
import time

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            )

loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-06
start_time = time.time()

for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print (i, loss.item())
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

elapsed_time = time.time() - start_time
print ("INFO: total elapsed time is : {}".format(elapsed_time))
    
