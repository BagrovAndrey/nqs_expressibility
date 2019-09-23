import torch
import torch.utils.data
import pickle
import sys
import numpy as np
import random_state_generator as rsg
import matplotlib.pyplot as plt

with open("./basis_1_N=20_k=10.dat", "rb") as input:
    loaded_vectors = pickle.load(input)
with open("./amplitudes_1_N=20_k=10.dat","rb") as input:
    loaded_amplitudes = pickle.load(input)

dim = len(np.binary_repr(loaded_vectors[-1]))

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = len(loaded_vectors), dim, 30, 1

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
#    torch.nn.Linear(H, H),
#    torch.nn.ReLU(),
#    torch.nn.Linear(H, H),
#    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
#    torch.nn.Softplus()
)

# Use the nn package to define our model and loss function.
shuffle = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_in),
)

shuffled_weights = shuffle.state_dict()
model_weights = model.state_dict()

#print(model_weights)

#with torch.no_grad():
#    torch.nn.init.uniform_(shuffled_weights["0.weight"],-1,1)
#    torch.nn.init.uniform_(shuffled_weights["2.weight"],-1,1)

#wing = 1

#with torch.no_grad():
#    torch.nn.init.uniform_(model_weights["0.weight"], -wing, wing)
#    torch.nn.init.uniform_(model_weights["2.weight"], -wing, wing)
#    torch.nn.init.uniform_(model_weights["4.weight"], -wing, wing)

y_pred = torch.squeeze(model(x))

loss_fn = torch.nn.MSELoss(reduction='sum')
print(loss_fn(y_pred, y))

with torch.no_grad():

    y_shuffled = torch.abs(torch.squeeze(model(x)))
    basis_order = torch.argsort(y_shuffled)

    norm = torch.norm(y_shuffled)
    shuffled_amplitudes = y_shuffled/norm
    x_coord = torch.arange(float(len(shuffled_amplitudes)))

    a_coeff = np.polyfit(x_coord, shuffled_amplitudes[basis_order], 1, rcond=None, full=False)[0]
    linear_fit = a_coeff*x_coord

    y = torch.empty(len(y_shuffled))
    y[basis_order] = linear_fit

    y = y/torch.norm(y)

    print(y)

fit_file = open("./linear_fit.dat", "wb")
pickle.dump(y.numpy(), fit_file)

sys.exit()

#plt.plot(x_coord, y[basis_order].detach().numpy())
#plt.plot(x_coord, shuffled_amplitudes[basis_order].detach().numpy())
#plt.show()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 3e-5
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y),
    batch_size=256,
    shuffle=True,
    )

y_pred = torch.abs(torch.squeeze(model(x)))
norm = torch.dot(y_pred,y_pred)
overlap = torch.dot(y, y_pred)
print(0, abs(overlap.item()/np.sqrt(norm.item())))

for t in range(10):

    for ibatch, (batch_x, batch_y) in enumerate(dataloader):
    
        y_pred = torch.abs(torch.squeeze(model(batch_x)))
#        loss = loss_fn(y_pred, batch_y)
        loss = 1 - torch.dot(y_pred, batch_y)/(torch.norm(y_pred)*torch.norm(batch_y))
  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = torch.abs(torch.squeeze(model(x)))

    if t % 1 == 0:
        norm = torch.dot(y_pred,y_pred)
        overlap = torch.dot(y, y_pred)
        print(t, abs(overlap.item()/np.sqrt(norm.item())))
        nqs_amplitudes_aux = y_pred/np.sqrt(norm.item())
        nqs_amplitudes = nqs_amplitudes_aux.detach().numpy()
        nqs_file = open("./NQS_amplitudes_"+str(t)+".dat", "wb")
        pickle.dump(nqs_amplitudes, nqs_file)

#sys.exit()

#shuffled_file = open("./NQS_amplitudes_9.dat", "rb")
#trained_amplitudes = pickle.load(shuffled_file)

plt.plot(x_coord, np.sort(y.detach().numpy()))
plt.plot(x_coord, np.sort(nqs_amplitudes))
plt.show()

"""    y_pred = torch.squeeze(model(x))
    loss = loss_fn(y_pred, y)

    if t % 1000 == 999:
        norm = torch.dot(y_pred,y_pred)
        overlap = torch.dot(y, y_pred)
        print(t, loss.item(), abs(overlap.item()/np.sqrt(norm.item())))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()"""
