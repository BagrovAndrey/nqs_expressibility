import numpy as np
import scipy.special
import pickle

import random_state_generator

import numba
from numba import uint64, float32


@numba.jit("uint64(uint64, uint64)", nogil=True, nopython=True)
def _binom(n, k):
    r"""Compute the number of ways to choose k elements out of a pile of n.

    :param n: the size of the pile of elements
    :param k: the number of elements to take from the pile
    :return: the number of ways to choose k elements out of a pile of n
    """
    assert 0 <= n and n < 40
    assert 0 <= k and k <= n

    if k == 0 or k == n:
        return 1
    total_ways = 1
    for i in range(min(k, n - k)):
        total_ways = total_ways * (n - i) // (i + 1)
    return total_ways


@numba.jit("uint64[:, :](uint64, uint64)", nogil=True, nopython=True)
def make_binom_cache(max_n, max_k):
    assert 0 < max_k and max_k <= max_n
    assert max_n < 40

    cache = np.zeros((max_n + 1, max_k + 2), dtype=np.uint64)
    for i in range(max_n + 1):
        for j in range(min(i, max_k) + 2):
            cache[i, j] = _binom(i, j) if j <= i else 0
    return cache


@numba.jit("uint64(uint64)", nogil=True, nopython=True)
def _hamming_weight(n: int):
    # See https://stackoverflow.com/a/9830282
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)
    return n


@numba.jit("uint64[:](uint64, uint64)", nogil=True, nopython=True)
def generate_binaries(length: int, hamming: int):
    assert length > 0
    assert hamming >= 0 and hamming <= length
    if hamming == 0:
        return np.zeros(1, dtype=np.uint64)

    size = _binom(length, hamming)
    set_of_vectors = np.empty(size, dtype=np.uint64)
    val = (1 << hamming) - 1
    for i in range(size):
        set_of_vectors[i] = val
        c = val & -val
        r = val + c
        val = (((r ^ val) >> 2) // c) | r
    return set_of_vectors


@numba.jit("uint64(uint64, uint64, uint64)", nogil=True, nopython=True)
def _merge(vec1, vec2, dim2):
    return (vec1 << dim2) | vec2


@numba.jit("uint64(uint64, uint64, uint64[:, :])", nogil=True, nopython=True)
def _get_index(x, dim, binom_cache):
    n = 0
    h = 0
    if x & 1 == 1:
        h += 1
    x >>= 1
    for i in range(1, dim):
        if x & 1 == 1:
            h += 1
            n += binom_cache[i, h]
        x >>= 1
    return n


@numba.jit(
    "float32(uint64, uint64, uint64, uint64, uint64, float32[::1], uint64[:, :])",
    nogil=True,
    nopython=True,
)
def _density_matrix_element(dim1, dim, hamming, vec1, vec2, amplitudes, binom_cache):
    hamming1 = _hamming_weight(vec1)
    smallest_number = 2 ** (hamming - hamming1) - 1
    k = dim - dim1
    index1 = _get_index(_merge(vec1, smallest_number, k), dim, binom_cache)
    index2 = _get_index(_merge(vec2, smallest_number, k), dim, binom_cache)
    size_of_traced = _binom(k, hamming - hamming1)
    matrix_element = np.dot(
        amplitudes[index1 : index1 + size_of_traced],
        amplitudes[index2 : index2 + size_of_traced],
    )
    return matrix_element


@numba.jit(
    "float32[:, :](uint64, uint64, uint64, uint64, float32[::1])",
    nogil=True,
    nopython=True,
    parallel=False,
)
def sector_density_matrix(sector_dim, dim, sector_hamming, hamming, amplitudes):
    assert sector_hamming <= hamming and sector_dim <= dim
    assert hamming - sector_hamming <= dim - sector_dim

    sector_basis = generate_binaries(sector_dim, sector_hamming)
    binom_cache = make_binom_cache(dim, hamming)
    n = len(sector_basis)
    matrix = np.empty((n, n), dtype=np.float32)
    for i in range(len(sector_basis)):
        for j in range(i, len(sector_basis)):
            matrix[i, j] = _density_matrix_element(
                sector_dim,
                dim,
                hamming,
                sector_basis[i],
                sector_basis[j],
                amplitudes,
                binom_cache,
            )
            matrix[j, i] = np.conj(matrix[i, j])

    return matrix


def density_matrix(sub_dim, dim, hamming, amplitudes):
    return [
        sector_density_matrix(sub_dim, dim, sub_hamming, hamming, amplitudes)
        for sub_hamming in range(
            max(0, hamming - (dim - sub_dim)), min(hamming, sub_dim) + 1
        )
    ]


def lambda_log_lambda(x):
    y = np.where(x > 1e-7, np.log(x), 0.0)
    y *= x
    return y


def main():
    with open("./basis_1_N=24_k=12.dat", "rb") as input:
        loaded_vectors = pickle.load(input)
    #with open("./amplitudes_1_N=24_k=12.dat", "rb") as input:
        #    with open("./linear_fit.dat","rb") as input:
    with open("./stacked_hom_sign.dat", "rb") as input:
        #    with open("./NQS_amplitudes_989.dat","rb") as input:
        loaded_amplitudes = pickle.load(input)

    scaling = []

    dim = 24
    hamming = dim // 2
    for sub_dim in range(1, dim // 2):
        rho = density_matrix(sub_dim, dim, hamming, loaded_amplitudes)
        entropy = 0

        for iloop in range(len(rho)):
            entang_spectrum = np.linalg.eigvalsh(rho[iloop])
            entropy -= lambda_log_lambda(entang_spectrum).sum()

        print(entropy, ",")

        scaling.append(entropy)

    print(scaling)


if __name__ == "__main__":
    main()
