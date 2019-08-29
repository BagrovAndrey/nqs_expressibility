import torch
import torch.utils.data
import pickle
import sys
import numpy as np
import random_state_generator as rsg

with open("./basis_1_N=20_k=10.dat", "rb") as input:
    loaded_vectors = pickle.load(input)
with open("./amplitudes_1_N=20_k=10.dat","rb") as input:
    loaded_amplitudes = pickle.load(input)

dim = len(np.binary_repr(loaded_vectors[-1]))

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = len(loaded_vectors), dim, 20, 1

binary_basis = np.array([rsg.spin2array(vec, dim) for vec in loaded_vectors])

x = torch.from_numpy(binary_basis)
y = torch.from_numpy(loaded_amplitudes)
y = y.squeeze()

# Define the data loader

dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y),
    batch_size=256,
    shuffle=True,
    )

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
#    torch.nn.Linear(H, H),
#    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

y_pred = torch.squeeze(model(x))

print(y_pred)

loss_fn = torch.nn.MSELoss(reduction='sum')
print(loss_fn(y_pred, y))

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 3e-5
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range(5000):

    for ibatch, (batch_x, batch_y) in enumerate(dataloader):
    
        y_pred = torch.squeeze(model(batch_x))
        loss = loss_fn(y_pred, batch_y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = torch.squeeze(model(x))
    if t % 100 == 99:
        norm = torch.dot(y_pred,y_pred)
        overlap = torch.dot(y, y_pred)
        print(t, abs(overlap.item()/np.sqrt(norm.item())))


"""    y_pred = torch.squeeze(model(x))
    loss = loss_fn(y_pred, y)

    if t % 1000 == 999:
        norm = torch.dot(y_pred,y_pred)
        overlap = torch.dot(y, y_pred)
        print(t, loss.item(), abs(overlap.item()/np.sqrt(norm.item())))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()"""
