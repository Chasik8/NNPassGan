import torch.nn as nn
import pytorch_lightning as pl
import torch
from model import *
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=0.5, std=0.5),
                                       transforms.Lambda(lambda x: x.view(-1, 784))])

data = datasets.MNIST(root='/data/MNIST', download=True, transform=mnist_transforms)

mnist_dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=4)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GAN()

trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, mnist_dataloader)