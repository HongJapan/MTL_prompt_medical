import torch
from PIL import Image
import glob
from torchvision.transforms import Compose, ColorJitter, RandomRotation

class Net(torch.nn.Module):

    def __init__(self, in_feat=512, out_feat=8):
        super(Net, self).__init__()
        #self.linear1 = torch.nn.Linear(in_feat, 128)
        self.linear2 = torch.nn.Linear(in_feat, out_feat)
        self.m = torch.nn.LogSoftmax(dim=1)
        #self.act = torch.nn.ReLU()

    def forward(self, X):
        # print(X.shape)
        # y = self.linear1(X)
        # y = self.act(y)
        X = self.m(self.linear2(X))
        return X