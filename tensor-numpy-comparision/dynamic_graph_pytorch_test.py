import torch
import random
import time

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-04, momentum = 0.9)
start_time = time.time()

for i in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print (i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

elapsed_time = time.time() - start_time
print ("INFO: Total elapsed time is : {}".format(elapsed_time))
