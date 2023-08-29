import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output

"""
Input
-----

A     : matrix
v     : vector
k     : integer
α,β   : recurrence coeffs.

Output
------
m     : modified moments to degree k for (α,β)
"""

def get_moments(A,v,k,α,β):
    
    m = np.full(k+1,np.nan)
    
    q = np.copy(v); q_ = np.zeros_like(v)
    m[0] = q.T@q

    for i in range(k):
        q__ = np.copy(q)
        q = 1 / β[i] * (A@q - α[i]*q - (β[i-1] if i>0 else 0)*q_) 
        q_ = q__

        m[i+1] = v.T@q

    return m


"""
Input
-----

A     : matrix
v     : vector
k     : integer
a,b   : endpoints

Output
------
m     : modified Cheb. moments to degree 2k
"""

def get_chebyshev_moments(A,v,k,a,b):
    
    m = np.full(2*k+1,np.nan)
    
    q = np.copy(v)
    m[0] = q.T@q
    
    q__ = np.copy(q)
    q = 2/(b-a) * (A@q - (a+b)/2*q)
    q_ = q__
    
    m[1] = np.sqrt(2)*q_.T@q

    for i in range(1,k):
        m[2*i] = np.sqrt(2)*(2*q.T@q - m[0])
        
        q__ = np.copy(q)
        q = 2 * 2/(b-a) * (A@q - (a+b)/2*q) - q_
        q_ = q__
        
        m[2*i+1] = np.sqrt(2)*(2*q_.T@q) - m[1]
        
    m[2*k] = np.sqrt(2)*(2*q.T@q - m[0])
    
    return m



"""
Input
-----

(α,β)   : recurrence coeffs.
(γ,δ)   : recurrence coeffs.

Output
------
C       : connection coefficient matrix (α,β) -> (γ,δ)
"""

def get_connection_coeffs(α,β,γ,δ):
    
#    α,β = M
#    γ,δ = N
    
    km1,km2 = len(α),len(β);
    kn1,kn2 = len(γ),len(δ);
    
    assert (km1==km2 or km1==km2+1) and (kn1==kn2 or kn1==kn2+1),\
           "Jacobi matrices must be size k,k or k+1,k"
    
    C = np.full((min(km2,kn2)+1,\
                 min(km2,kn1+kn2)+1)\
                ,np.nan)
    
    C[0,0] = 1
    for j in range(1,min(km2,kn1+kn2)+1):
        for i in range(0,min(j,kn1+kn2-j)+1):
            C[i,j] =( ( δ[i-1]*C[i-1,j-1]         if i-1>=0 else 0)\
                      +( (γ[i]-α[j-1])*C[i,j-1]    if i<=j-1 else 0)\
                      +( δ[i]*C[i+1,j-1]           if i+1<=j-1 else 0)\
                      -( β[j-2]*C[i,j-2]           if i<=j-2 else 0)\
                     )/β[j-1]    

    return C

"""
Input
-----

A     : matrix
v     : vector
s     : integer
α,β   : recurrence coeffs.
a,b   : endpoints
m     : modified Cheb. moments to degree k

Output
------
m     : modified moments to degree s for (α,β)
"""

def get_moments_from_cheb(A,v,s,α,β,a,b,m=None):
    
    k = int(np.ceil(s/2))

    if m is None:
        n = get_chebyshev_moments(A,v,k,a,b)
    else:
        n = m
        
    γ = np.ones(s)*(a+b)/2
    δ = np.ones(s)*(b-a)/4
    δ[0] *= np.sqrt(2)

    C = get_connection_coeffs(α[:2*k],β[:2*k],γ,δ)
        
    return np.triu(C).T[:s+1,:s+1]@n[:s+1]

"""
Input
-----

A       : matrix
v       : vector
s       : integer
α,β     : recurrence coeffs.
T=(γ,δ) : Jacobi coefficients for Ψ 

Output
------
m     : modified moments to degree s for (α,β)
"""

def get_moments_from_lanczos(A,v,s,α,β,T=None):
    
    k = int(np.ceil(s/2))
    if T is None:
        (γ,δ) = lanczos(A,v,k)
    else:
        (γ,δ) = T
        
    C = get_connection_coeffs(α[:2*k],β[:2*k],γ,δ)

    return C[0]

"""
Input
-----

m       : matrix
s       : integer
α,β     : recurrence coeffs.

Output
------
θ,ω     : nodes and weights for degree s quadrature approximation 
"""

def get_iq(m,s,α,β):
    
    try:
        θ,S = sp.linalg.eigh_tridiagonal(α[:s+1],β[:s])
    except:
        M = np.diag(α[:s+1]) + np.diag(β[:s],1) + np.diag(β[:s],-1)
        θ,S = sp.linalg.eigh(M)
    
    ω = S[0]*(S.T@m[:s+1])
    return θ,ω

"""
Input
-----

k       : integer
α,β     : Jacobi coefficients for Ψ 

Output
------
θ,ω     : nodes and weights for degree 2k-1 Gaussian quadrature approximation 
"""

def get_gq(k,α,β):
    
    try:
        θ,S = sp.linalg.eigh_tridiagonal(α[:k],β[:k-1])
    except:
        T = np.diag(α[:k]) + np.diag(β[:k-1],1) + np.diag(β[:k-1],-1)
        θ,S = sp.linalg.eigh(T)

    ω = np.abs(S[0])**2
    
    return θ,ω

# get_iaq(m,s,s+1,αT,βT) is same as IQ
def get_aaq(m,s,d,α,β):
    
    assert d>s, 'need at d at least s+1'
        
    try:
        θ,S = sp.linalg.eigh_tridiagonal(α[:d],β[:d-1])
    except:
        T = np.diag(α[:d]) + np.diag(β[:d-1],1) + np.diag(β[:d-1],-1)
        θ,S = sp.linalg.eigh(T)

    ω = S[0]*(S[:s+1,:].T@m[:s+1])
    
    return θ,ω

def eval_poly(x,m,α,β):
    
    s = len(m)-1
    
    p = np.zeros_like(x)
    q = np.ones_like(x); q_ = np.zeros_like(x)
    p += m[0]
    for i in range(s):
        q__ = np.copy(q)
        q = 1 / β[i] * (x*q - α[i]*q - (β[i-1] if i>0 else 0)*q_) 
        q_ = q__

        p += m[i+1] * q
    
    return p

def jackson_weights(k):
    return (1/(k+1))*((k-np.arange(k)+1)*np.cos(np.pi*np.arange(k)/(k+1))+np.sin(np.pi*np.arange(k)/(k+1))/np.tan(np.pi/(k+1)))

def get_op_recurrence(intervals,weights,α,β,k):
    
    l = len(weights)
    interval_widths = intervals[:,1] - intervals[:,0]
    interval_centers = (intervals[:,1] + intervals[:,0])/2

    GQ_nodes,GQ_weights = get_gq(k,α,β)
    
    #GQ_nodes = np.cos((np.arange(k)+1)/(k+1)*np.pi)
    #GQ_weights = np.pi/(k+1)*np.sin((np.arange(k)+1)/(k+1)*np.pi)**2/np.pi*2#np.ones(k)/k
    
    
    lam = np.kron(interval_widths/2,GQ_nodes) + np.kron(interval_centers,np.ones(k))
    A = sp.sparse.spdiags(lam,0,(k)*l,(k)*l)
    v = np.kron(np.sqrt(weights),np.sqrt(GQ_weights))
    
    (α1,β1) = lanczos_reorth(A,v,k+1,reorth=True)
    
    return α1[:k],β1[:k]

def get_multi_chebyshev_recurrence(intervals,weights,k):
    
    l = len(weights)
    interval_widths = intervals[:,1] - intervals[:,0]
    interval_centers = (intervals[:,1] + intervals[:,0])/2

    GQ_nodes = np.cos((2*(np.arange(k)+1)-1)/(2*k)*np.pi)
    GQ_weights = np.ones(k)/k
    
    lam = np.kron(interval_widths/2,GQ_nodes) + np.kron(interval_centers,np.ones(k))
    A = sp.sparse.spdiags(lam,0,k*l,k*l)
    v = np.kron(np.sqrt(weights),np.sqrt(GQ_weights))
    
    (α,β) = lanczos(A,v,k+1)
    
    return α,β