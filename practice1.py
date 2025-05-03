import torch
from torch import nn 
import matplotlib as plt
import numpy as np

weight =0.7
bias = 0.3

start=0
end=1
step=0.02

s1=torch.arange(start,end,step)

s2=weight*s1 + bias

splitint=int(0.8*len(s1))

testx,testy=s1[splitint:],s2[splitint:]
trainx,trainy=s1[:splitint],s2[:splitint]

def plotdagraph(traindatax=trainx,
                traindatay=trainy,
                testdatax=testx,
                testdatay=testy,
                prediction=None):
    plt.figure(figsize=(10,7))#read doc for this 

    plt.scatter(traindatax,traindatay,c="b",s=4,label="training data")

    plt.scatter(testdatax,testdatay,c="r",s=4,label="testing data")

    if prediction is not None:
        plt.scatter(testdatax,prediction,c="g",s=4,label="predictions")
    
    plt.legend(prop={"size": 14})

class niggamlmodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    
    def forward(self, x:torch.tensor)-> torch.Tensor:
        return self.weights*x + self.bias

torch.manual_seed(69)

prototype1=niggamlmodel()
print(list(niggamlmodel.parameters()))

