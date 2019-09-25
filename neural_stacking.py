import torch
import torch.utils.data
import pickle
import sys
import numpy as np
import random_state_generator as rsg
import matplotlib.pyplot as plt

with open("./basis_1_N=24_k=12.dat", "rb") as input:
    loaded_vectors = pickle.load(input)
with open("./amplitudes_1_N=24_k=12.dat","rb") as input:
    loaded_amplitudes = pickle.load(input)

dim = len(np.binary_repr(loaded_vectors[-1]))

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

N, D_in, H, D_out = len(loaded_vectors), dim, 30, 1

binary_basis = np.array([rsg.spin2array(vec, dim) for vec in loaded_vectors])
x = torch.from_numpy(binary_basis)

# This function generates a stack of neural networks that would be further combined into uncorrelated states 

def neural_stack(amp_depth, sign_depth):

    amp_model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )

    amp_initial = (torch.abs(torch.squeeze(amp_model(x)))).detach().numpy()

    sign_model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )

    sign_blueprint_initial = (torch.abs(torch.squeeze(sign_model(x)))).detach().numpy()

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

            aux_amp = [(torch.abs(torch.squeeze(amp_model(x)))).detach().numpy()]
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

            aux_sign = [(torch.abs(torch.squeeze(sign_model(x)))).detach().numpy()]
            sign_blueprint_stack = np.concatenate((sign_blueprint_stack, aux_sign), axis=0)

    return(amplitude_stack, sign_blueprint_stack)

evaluated_stack = neural_stack(6,6) # Generating stacks
amplitude_aux = np.prod(evaluated_stack[0], axis = 0)
amplitude_vector = amplitude_aux/np.linalg.norm(amplitude_aux)  # Resulting vector of amplitudes
sign_aux = np.prod(evaluated_stack[1], axis = 0)
sign_blueprint = sign_aux/np.linalg.norm(sign_aux) # Resulting vector that can be used to generate signs

# full_sign_vec = 1 - 2 * np.random.choice([0, 1], size=len(amplitude_vector))

# Generating sign structure
full_sign_vec = np.sign(sign_blueprint - np.median(sign_blueprint))

# Generating signful quantum state
full_vector = np.multiply(amplitude_vector, full_sign_vec)

# Saving positive quantum state

amp_file = open("./stacked_amp.dat", "wb")
pickle.dump(amplitude_vector, amp_file)

# Saving signful quantum state

full_file = open("./stacked_full.dat", "wb")
pickle.dump(full_vector, full_file)
