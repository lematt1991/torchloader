# torchloader
Use PyTorch style dataloaders without any PyTorch dependencies.

The [`Dataset`](http://pytorch.org/docs/0.3.0/data.html) model of loading data into a neural network is very convienient, especially when generalizing the process to multiple processes.  This package brings this type of simplicity to other deep learning frameworks, or any other use case for that matter.

# Install

```
pip install torchloader
```

# Usage


```Python

from torchloader import Dataset, DataLoader
import cv2

class MyDataset(Dataset):
	def __init__(self, filenames):
		self.filenames = filenames
	
	def __len__(self):
		# return the length of the dataset
		return len(self.filenames)
	
	def __get_item__(self, idx):
		file = self.filenames[idx]
		return cv2.imread(file)

dataset = MyDataset(glob.glob(...))
loader = DataLoader(dataset, batch_size=16, num_workers=4)

for inputs in loader:
	...

```

See the original [PyTorch docs](http://pytorch.org/docs/0.3.0/data.html) for additional information