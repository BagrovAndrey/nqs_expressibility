import numpy as np
import scipy.special
import pickle
import matplotlib.pyplot as plt

def generate_binaries(length, hamming):

    if length>0 and hamming >=0 and hamming <=length:
        smallest_number = sum([2**(jloop) for jloop in range(hamming)])
    else:
        print("Wrong Hamming weight or length")
        sys.exit

    set_of_vectors = []
    set_of_vectors.append(smallest_number)

    if hamming > 0:

        next = 0
        val = smallest_number

        while next < 2**length:
            c = val & -val;
            r = val + c;
            next = (((r^val) >> 2) // c) | r
            val = next
            set_of_vectors.append(next)

        del set_of_vectors[-1]

    return(set_of_vectors)

def generate_amplitude(dimension):
    set_of_amplitudes = np.random.uniform(-1, 1, size = dimension).astype(np.float32)
    normalization = np.sqrt(np.einsum('i,i', set_of_amplitudes, set_of_amplitudes))
    set_of_amplitudes = set_of_amplitudes / normalization

#    return np.sort(set_of_amplitudes)
    return set_of_amplitudes

def conserve(vectors, amplitudes, length, hamming, name_id):
 
    vector_file = open("basis_"+str(name_id)+"_N="+str(length)+"_k="+str(hamming)+".dat", "wb")
    amplitude_file = open("amplitudes_"+str(name_id)+"_N="+str(length)+"_k="+str(hamming)+".dat", "wb")
    pickle.dump(vectors, vector_file)
    pickle.dump(amplitudes, amplitude_file)

def conserve_dict(dict_psi, length, hamming, name_id):
    
    dict_file = open("psi_"+str(name_id)+"_N="+str(length)+"_k="+str(hamming)+".dat", "wb")
    pickle.dump(dict_psi, dict_file)

def spin2array(n, size):
    s = '{0:0{1:}b}'.format(n, size)
    return np.fromiter((2 * float(c) - 1 for c in s), dtype=np.float32, count=size)

def main():

    print(generate_binaries(10,0))

    name_id = 1
    length = 24
    hamming = 12
    dimension = int(scipy.special.binom(length, hamming))

    vectors = generate_binaries(length, hamming)
    amplitudes = generate_amplitude(dimension)
    
# A uniform vector of amplitudes - to test the role of sign structure

#    amplitudes = np.ones(dimension).astype(np.float32)
#    normalization = np.sqrt(np.einsum('i,i', amplitudes, amplitudes))
#    amplitudes = amplitudes / normalization

    #signs = np.sign(np.random.uniform(-1, 1, size = dimension).astype(np.float32))
    #amplitudes = np.einsum('i,i->i', amplitudes, signs)

    print(len(amplitudes), amplitudes)

#    plt.plot(amplitudes)
#    plt.show()

    zipped_psi = zip(vectors, amplitudes)
    dict_psi = dict(zipped_psi)

    conserve(vectors, amplitudes, length, hamming, name_id)
    conserve_dict(dict_psi, length, hamming, name_id)

if __name__ == "__main__":
    main()
