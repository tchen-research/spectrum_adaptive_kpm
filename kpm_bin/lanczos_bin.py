import numpy as np
import scipy as sp


def lanczos(A,q0,k):
    """
    run Lanczos with reorthogonalization
    
    Input
    -----
    A : entries of diagonal matrix A
    q0 : starting vector
    k : number of iterations
    """
    
    n = len(q0)
    
    α = np.zeros(k,dtype=A.dtype)
    β = np.zeros(k,dtype=A.dtype)
    
    qi = q0 / np.linalg.norm(q0)
    for i in range(1,k+1):
        # expand Krylov space

        qim1 = np.copy(qi)
        qi = A@qim1 - β[i-2]*qim2 if i>1 else A@qim1
        qim2 = np.copy(qim1)
        
        α[i-1] = qim1.conj().T@qi
        qi -= α[i-1]*qim1

            
        β[i-1] = np.linalg.norm(qi)
        qi = qi / β[i-1]

    return (α,β)

def lanczos_reorth(A,v,k,reorth=True,returnQ=False):
    
    n = len(v)
    
    Q = np.zeros((n,k),dtype=v.dtype)
    α = np.zeros(k,dtype=np.float64)
    β = np.zeros(k,dtype=np.float64)
    
    Q[:,0] = v / np.sqrt(v.T@v)
    q = Q[:,0]
    for i in range(0,k):
        # expand Krylov space
        
        q = A@Q[:,i] - β[i-1]*Q[:,i-1] if i>0 else A@Q[:,i]
       
        α[i] = q@Q[:,i].astype('float64') # for some reason the inner product not exactly equivalent otherwise.
        q -= α[i]*Q[:,i]
        
        if reorth:
            q -= Q@(Q.T@q) # regular GS
            #for j in range(i-1): # modified GS (a bit too slow)
            #    q -= (q.T@Q[:,j])*Q[:,j]
        
        β[i] = np.sqrt(q.T@q)    
        if i < k-1:
            Q[:,i+1] = q / β[i]
            q = Q[:,i+1]
    
    if returnQ:
        return Q,(α,β)
    else:
        return (α,β)
