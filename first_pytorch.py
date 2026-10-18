import torch
import torch.nn as nn
import torch.optim as optim

import helper_utils

torch.manual_seed(42)

distances = torch.tensor([[1.0],[2.0],[3.0],[4.0]], dtype = torch.float32)
times = torch.tensor([[6.96],[12.11],[16.77],[22.21]], dtype=torch.float32)

model = nn.Sequential(nn.Linear(1, 1))

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(distances)

    loss = loss_function(outputs, times)

    loss.backward()

    optimizer.step()

    if (epoch+1)%10 == 0:
        print(f'epoch {epoch+1}, loss {loss.item():.4f}')

x = torch.tensor([2.0, 4.0, 6.0])
y = torch.tensor([1.0, 2.0, 3.0])
result = x / y + 5
print(result)

x = torch.tensor([25.0])
x = x.unsqueeze(0)
x = x.squeeze()
print(x.shape)
    
    