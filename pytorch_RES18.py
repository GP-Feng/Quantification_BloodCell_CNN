import torch.nn as nn
from torchvision import models

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.base_model.fc = nn.Sequential(nn.Linear(512,1),
                                 nn.Sigmoid())


    def forward(self, input):
        outputs = self.base_model(input)
        return outputs