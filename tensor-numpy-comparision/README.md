## Notes on comparision and similarities of numpy and pytorch tensors

* Tensor is identical to numpy array.
* Pytorch tensor is implemented to speedup the operations either on GPU or CPU.
* Backward gradient computation is automatically handled by autograd package. Doing loss.backward() calculates all the gradients.
* To create a custom autograd function, we implement the class as a subfunction to torch.autograd.Function and implementing the static methods -> forward and backward.
* In the above examples here is the analysis for 500 iterations on a CPU:

Test file name | Time 
------| ----------
numpy_test.py | 9 sec 
pytorch_tensor_test.py | 1.5 sec 
tensor_autograd_test.py | 0.54 sec |
custom_autograd_test.py | 0.84 sec
