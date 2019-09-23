import numpy as np
import scipy.special
import pickle
import random_state_generator
import matplotlib

def vector_merger(vec1, vec2, dim1, dim2):

    binary1 = np.binary_repr(vec1, dim1)
    binary2 = np.binary_repr(vec2, dim2)

    return int(binary1+binary2,2)

def density_matrix_element(sub_dim, dim, vec1, vec2, basis, amplitudes):

    binary1 = np.binary_repr(vec1, sub_dim)
    binary2 = np.binary_repr(vec2, sub_dim)

    if binary1.count('1') != binary2.count('1'):

        print("Magnetizations do not match")

    sub_magnetization = binary1.count('1')

    magnetization = np.binary_repr(basis[0]).count('1')

    smallest_number = sum([2**(jloop) for jloop in range(magnetization - sub_magnetization)])
    first_state_1 = vector_merger(vec1, smallest_number, sub_dim, dim - sub_dim)
    first_state_2 = vector_merger(vec2, smallest_number, sub_dim, dim - sub_dim)

    try:
        index_1 = basis.index(first_state_1)
    except ValueError as e:
        print(sub_magnetization, magnetization, vec1, smallest_number)
        raise e
    index_2 = basis.index(first_state_2)

    size_of_traced = int(scipy.special.binom(dim - sub_dim, magnetization - sub_magnetization))

    matrix_element = np.sum(amplitudes[index_1:index_1 + size_of_traced] * amplitudes[index_2:index_2 + size_of_traced])

#    for iloop in range(size_of_traced):
#        matrix_element = matrix_element + amplitudes[iloop+index_1]*amplitudes[iloop+index_2]    

    return matrix_element

def density_matrix(sub_dim, sub_magnetization, basis, amplitudes):

    dim = len(np.binary_repr(basis[-1]))
    magnetization = (np.binary_repr(basis[-1])).count('1')

    if sub_magnetization > magnetization or sub_magnetization > sub_dim or (magnetization - sub_magnetization) > (dim - sub_dim):
        print("Too large sub-magnetization")

    sub_sector = random_state_generator.generate_binaries(sub_dim, sub_magnetization)

    sector_matrix = [[density_matrix_element(sub_dim, dim, sub_sector[iloop], sub_sector[jloop], basis, amplitudes) for iloop in range(len(sub_sector))] for jloop in range(len(sub_sector))]
    sector_matrix = np.array(sector_matrix)

    return sector_matrix

def full_density_matrix(sub_dim, basis, amplitudes):

    dim = len(np.binary_repr(basis[-1]))
    magnetization = (np.binary_repr(basis[-1])).count('1')
    rhoA = []
    
    for kloop in range(max(0, magnetization - (dim - sub_dim)), min(magnetization, sub_dim) + 1):
        sector_matrix = density_matrix(sub_dim, kloop, basis, amplitudes)
        rhoA.append(sector_matrix)

    return rhoA

def lambda_log_lambda(x):
    
    if x > 1e-7:
        y = x*np.log(x)
    else:
        y = 0
    
    return y

def main():
    with open("./basis_1_N=20_k=10.dat", "rb") as input:
        loaded_vectors = pickle.load(input)
#    with open("./amplitudes_1_N=20_k=10.dat","rb") as input:
#    with open("./linear_fit.dat","rb") as input:
    with open("./NQS_amplitudes_459.dat","rb") as input:
        loaded_amplitudes = pickle.load(input)

    scaling = []

    for sub_dim in range(1,10):

        rho = full_density_matrix(sub_dim, loaded_vectors, loaded_amplitudes)
        x = 0
        entropy = 0

        for iloop in range(len(rho)):
            x = x + np.einsum('ii', rho[iloop])
            entang_spectrum = np.linalg.eig(rho[iloop])[0]
            entropy = entropy - sum(map(lambda_log_lambda, entang_spectrum))

        print(entropy,",")
         
        scaling.append(entropy)

    print(scaling)

if __name__ == "__main__":
    main()
