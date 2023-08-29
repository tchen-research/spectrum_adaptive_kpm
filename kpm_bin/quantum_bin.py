import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from itertools import product

def pauli(s,dir):
    M = int(2*s+1)

    if dir =='y':
        S = np.zeros((M,M),dtype='complex')
    else:
        S = np.zeros((M,M))

    for i in range(M):
        for j in range(M):

            if dir == 'x':
                S[i,j] = ((i==j+1)+(i+1==j))*np.sqrt(s*(s+1)-(s-i)*(s-j))
            if dir == 'y':
                S[i,j] = ((i+1==j)-(i==j+1))*np.sqrt(s*(s+1)-(s-i)*(s-j))/1j

            if dir == 'z':
                S[i,j] = (i==j)*(s-i)*2
    if dir == 'I':
        return np.eye(M)

    return S
    
def kron_list(M):
    """
    return kroneker product of list of matrices
    """
    
    if len(M) == 1:
        return M[0]
    else:
        return sp.sparse.kron(M[0],kron_list(M[1:]),format='coo')

def get_hamiltonian(Jx,Jy,Jz,h,s):
    """
    General Hamiltoniam mxatrix for XYZ spin system with isotropic magnetic field in z direction
    """
    
    N = len(Jx)

    M = int(2*s+1)
    Sx = pauli(s,'x')
    Sy = pauli(s,'y')
    Sz = pauli(s,'z')
    
    out = sp.sparse.coo_matrix((M**N,M**N))

    for j in range(N):
        for i in range(j):

            I1 = sp.sparse.eye(M**(i))
            I2 = sp.sparse.eye(M**(j-i-1))
            I3 = sp.sparse.eye(M**(N-j-1))

            if Jx[i,j] != 0:
                Sxi_Sxj = kron_list([I1,Sx,I2,Sx,I3])
                out += 2 * Jx[i,j] * Sxi_Sxj

            if Jy[i,j] != 0:
                Syi_Syj = kron_list([I1,Sy,I2,Sy,I3])
                out += 2 * Jy[i,j] * np.real(Syi_Syj)

            if Jz[i,j] != 0:
                Szi_Szj = kron_list([I1,Sz,I2,Sz,I3])
                out += 2 * Jz[i,j] * Szi_Szj

        I1 = sp.sparse.eye(M**j)
        I2 = sp.sparse.eye(M**(N-j-1))

        Sxi_Sxi =  kron_list([I1,Sx@Sx,I2])
        Syi_Syi =  kron_list([I1,np.real(Sy@Sy),I2])
        Szi_Szi =  kron_list([I1,Sz@Sz,I2])

        out += Jx[j,j] * Sxi_Sxi
        out += Jy[j,j] * Syi_Syi
        out += Jz[j,j] * Szi_Szi

        Szj = kron_list([I1,Sz,I2])
        out += h * Szj

    return out.tocsr()
        
def get_solvable_density_EVs(h,J,N):
    
    k = np.arange(N)+1
    
    λk = h - 2*J*np.cos(k*np.pi/(N+1))
    
    Evs = np.full(2**N,np.nan)
    
    n_comb = [i for i in product(range(2), repeat=N)]
    
    for i,n in enumerate(n_comb):
        Evs[i] = n @ λk - N*h/2
    
    return np.sort(Evs)

