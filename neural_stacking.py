import torch
import torch.utils.data
import pickle
import sys
import numpy as np
import random_state_generator as rsg
# import matplotlib.pyplot as plt

with open("./basis_1_N=24_k=12.dat", "rb") as input:
    loaded_vectors = pickle.load(input)
with open("./amplitudes_1_N=24_k=12.dat","rb") as input:
    loaded_amplitudes = pickle.load(input)

dim = len(np.binary_repr(loaded_vectors[-1]))

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = len(loaded_vectors), dim, 32, 1

binary_basis = np.array([rsg.spin2array(vec, dim) for vec in loaded_vectors])
x = torch.from_numpy(binary_basis)

# This function generates a stack of neural networks that would be further combined into uncorrelated states 

def neural_stack(amp_depth, sign_depth, power1, power2):

    amp_model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )

    indices = torch.randperm(x.size(1))
    permute_x = x[:, indices]

    amp_initial = ((torch.abs(torch.squeeze(amp_model(x)))).detach().numpy())**power1
    permute_initial = ((torch.abs(torch.squeeze(amp_model(permute_x)))).detach().numpy())**power1

    sign_model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )

    sign_blueprint_initial = ((torch.abs(torch.squeeze(sign_model(x)))).detach().numpy())**power2

    amplitude_stack = [amp_initial]
    sign_blueprint_stack = [sign_blueprint_initial]

# Generates a stack of networks responsible for amplitudes

    if amp_depth > 1:

        for iloop in range(amp_depth - 1):

            print(iloop)

            amp_model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            )

            aux_amp = [((torch.abs(torch.squeeze(amp_model(x)))).detach().numpy())**power1]
            amplitude_stack = np.concatenate((amplitude_stack, aux_amp), axis=0)

# Generates a stack of networks responsible for signs

    if sign_depth > 1:

        for iloop in range(sign_depth - 1):

            sign_model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
            )

            aux_sign = [((torch.abs(torch.squeeze(sign_model(x)))).detach().numpy())**power2]
            sign_blueprint_stack = np.concatenate((sign_blueprint_stack, aux_sign), axis=0)

    return amplitude_stack, sign_blueprint_stack, permute_initial

evaluated_stack = neural_stack(5,10,1/15.,1/30.) # Generating stacks

amplitude_aux = np.prod(evaluated_stack[0], axis = 0)
amplitude_vector = amplitude_aux/np.linalg.norm(amplitude_aux)  # Resulting vector of amplitudes

permute_aux = evaluated_stack[2]
permute_vector = permute_aux/np.linalg.norm(permute_aux)  # Resulting vector of permuted amplitudes

sign_aux = np.prod(evaluated_stack[1], axis = 0)
sign_blueprint = sign_aux/np.linalg.norm(sign_aux) # Resulting vector that can be used to generate signs

# full_sign_vec = 1 - 2 * np.random.choice([0, 1], size=len(amplitude_vector))

# Generating sign structure
full_sign_vec = np.sign(sign_blueprint - np.median(sign_blueprint))
rand_sign_vec = np.sign(np.random.uniform(-1, 1, size = len(binary_basis)).astype(np.float32))

# Generating signful quantum state
full_vector = np.multiply(amplitude_vector, full_sign_vec)
rand_vector = np.multiply(amplitude_vector, rand_sign_vec)

# Saving positive quantum state

amp_file = open("./stacked_amp.dat", "wb")
pickle.dump(amplitude_vector, amp_file)

perm_file = open("./stacked_perm.dat", "wb")
pickle.dump(permute_vector, perm_file)

# Saving signful quantum state

full_file = open("./stacked_full.dat", "wb")
pickle.dump(full_vector, full_file)

rand_file = open("./stacked_rand.dat", "wb")
pickle.dump(rand_vector, rand_file)

# Saving sorted distribution

# sort_vector = np.sort(amplitude_vector)
# sort_file = open("./stacked_sort.dat", "w")
# for item in sort_vector:
#     sort_file.write(str(item))
