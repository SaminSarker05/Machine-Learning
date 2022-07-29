import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class neural(nn.Module):
    def __init__(self, input, hidden1, hidden2, classes):
        super(neural,self).__init__()
        self.l1 = nn.Linear(input,hidden1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden2, classes)
    def forward(self,x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x

# layer, nodes, classes
model = neural(4, 100, 50, 3)

# getting data
trainload = nn.utils.data.DataLoader(dataset=train, batch_size=10, shuffle=True)
testload = nn.utils.data.DataLoader(dataset=test, batch_size=10, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1) # updates weights using SGD
criterion = nn.MSELoss() # loss function

c, total = 0,0

for epoch in range(5):
    for i, (x,y) in enumerate(trainload):
        yhat = model(x)
        loss = criterion(yhat,y)
        loss.backward()
        opimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        for x,y in testload:
            y0 = torch.max(model(x))
            total += 1
            c += (y0 == y)
        accuracy = c/total * 100
        c, total = 0,0