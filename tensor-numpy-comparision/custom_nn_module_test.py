import torch
import time

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-06)
start_time = time.time()

for i in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print (i, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

elapsed_time = time.time() - start_time
print ("INFO: total time elapsed is : {}".format(elapsed_time))

