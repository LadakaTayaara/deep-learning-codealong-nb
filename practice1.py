#import matplotlib and pytorch

import torch
from torch import nn 
import matplotlib.pyplot as plt  # Fixed import
from pathlib import Path

#check pytorch version 
print(torch.__version__)

#create device agnostic code , i.e code will use GPU for faster computing 
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"device using {device}")

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

    plt.scatter(testdatax,testdatay,c="g",s=4,label="testing data")

    if prediction is not None:
        plt.scatter(testdatax,prediction,c="r",s=4,label="predictions")
    
    plt.legend(prop={"size": 14})

class niggamlmodel(nn.Module):
    def __init__(self):
        super().__init__()
        #use nn.linear() for creating the model parameters / also called linear transform , probing layer , fully connected layer
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1) #take input of size 1 and output of size 1
         
    def forward(self, x:torch.tensor)-> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(69)

prototype1=niggamlmodel()

#set the model to use the target device
prototype1.to(device)

#for training we need loss function , optimizer , training loop and testing loop 
#loss function
loss_fn=nn.L1Loss()
#setup optimizer
optimizer=torch.optim.SGD(params=prototype1.parameters(),
                          lr=0.01,)

#training loop 
torch.manual_seed(69)

epochs=200

#put data on the target device and reshape to match model input (N, 1)
testx=testx.unsqueeze(1).to(device)
testy=testy.to(device)
trainx=trainx.unsqueeze(1).to(device)
trainy=trainy.to(device)

for epoch in range(epochs):
    prototype1.train()
    #forward pass to get y
    y_pred=prototype1(trainx)

    #calculate loss
    loss=loss_fn(y_pred,trainy.unsqueeze(1))  # Match shapes with model output, format:- input and the targetted value

    #optimizer zero grad
    optimizer.zero_grad()

    #perform back prop
    loss.backward()

    #optimizer step
    optimizer.step()

    #Testing
    prototype1.eval()
    with torch.inference_mode():
        test_pred=prototype1(testx)
        test_loss=loss_fn(test_pred,testy.unsqueeze(1))  # Match shapes

    if epoch%10 ==0 :
        print(f"epoch {epoch} loss {loss} test loss {test_loss}")

print(prototype1.state_dict())
print(weight)
print(bias)

prototype1.eval()

#make prediction on the test data 
with torch.inference_mode():
    y_preds=prototype1(testx)

#numpy is CPU based , and we need to pass numpy array in plotgraph
plotdagraph(prediction=y_preds.cpu())
plt.show()










#create model save path 
model_name="type_01_1_prototype"
model_save_path=Path("modlstest")
model_save_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
model_save_path = model_save_path / model_name

#save the model 
torch.save(obj=prototype1.state_dict(),
           f=model_save_path)

loaded_model_=niggamlmodel()
loaded_model_.load_state_dict(torch.load(model_save_path))
loaded_model_.to(device)

print(loaded_model_.state_dict())

#evaluate model 
loaded_model_.eval()
with torch.inference_mode():
    loaded_model_1_preds=loaded_model_(testx)
print(y_preds==loaded_model_1_preds)