import torch
import time

N, D_in, H, D_out = 640, 4096, 2048, 1024

model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(H, D_out),
                            torch.nn.Dropout(p=0.2)).cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

static_input = torch.randn(N, D_in, device='cuda')
static_target = torch.randn(N, D_out, device='cuda')


def run_in_eager():
    optimizer.zero_grad(set_to_none=True)
    y_pred = model(static_input)
    loss = loss_fn(y_pred, static_target)
    loss.backward()
    optimizer.step()

## eager warmup.
run_in_eager()
run_in_eager()

import time
torch.cuda.synchronize()
start_time = time.time()
for j in range(10):
    run_in_eager()
torch.cuda.synchronize()
end_time = time.time()

print ("Time in eager: {} s".format(end_time - start_time))


## warmup cuda graphs.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    for i in range(3):
        run_in_eager()
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

start_time = time.time()
for data, target in zip(real_inputs, real_targets):
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()

end_time = time.time()

print ("Time in CUDA graphs with eager model: {}s".format(end_time - start_time))


#### Torch.compile with cudagraphs.

model_compile = torch.compile(model, mode="reduce-overhead")

## warmup
def run_in_compile():
    optimizer.zero_grad(set_to_none=True)
    y_pred = model_compile(static_input)
    loss = loss_fn(y_pred, static_target)
    loss.backward()
    optimizer.step()
 
for i in range(3):
    run_in_compile()


torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    run_in_compile()
end_time = time.time()

print ("Time in compile : {}s".format(end_time - start_time))


## CUDAGraphs + torch.compiled model.


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    for i in range(3):
        run_in_compile()
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model_compile(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

start_time = time.time()
for data, target in zip(real_inputs, real_targets):
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()

end_time = time.time()

print ("Time in CUDA graphs with compile model: {}s".format(end_time - start_time))
