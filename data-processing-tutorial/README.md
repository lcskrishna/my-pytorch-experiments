### Notes on Data pre-processing for a custom dataset

* Custom dataset inherits the Dataset class and overrides the methods ```__init__``` and ```__getitem__```
* Sample of the dataset is a dict of format ```{'image': image, 'label':label}```
* init function takes the directory of images, labels and transform that is used for further transformation of data.
* Transformations can be done as a seperate classes which needs to implement ```__call__``` or ```__init__``` methods.
* Compose the transformations using torchvision.tranforms.Compose
* All the datasets can be composed in a single line using Compose.
* To create a batch of images in batches, use the torch.utils.data.DataLoader API.
