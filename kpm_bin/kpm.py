import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output
from .lanczos_bin import *

"""
Input
-----

A     : matrix
v     : vector
k     : integer
γ,δ   : recurrence coeffs.

Output
------
m     : modified moments to degree k for (γ,δ)
"""

def get_cheb_moments(A,v,k,Emin,Emax):
    a = (Emax+Emin)/2
    b = (Emax-Emin)/2

    μ = np.full(2*k,np.nan)

    q_ = np.copy(v)
    q = (A@q_-a*q_)/b
    
    μ[0] = 1
    μ[1] = np.sqrt(2)*q@q_

    for n in range(1,k):
        q__ = np.copy(q)
        q = 2*(A@q - a*q)/b - q_
        q_ = q__
        
        μ[2*n] = np.sqrt(2)*(2*q_@q_-μ[0])
        μ[2*n+1] = (np.sqrt(2)*2*q@q_-μ[1])

    return μ


def get_moments(A,v,k,γ,δ):
    
    μ = np.full(k,np.nan)
    
    q = np.copy(v); q_ = np.zeros_like(v)
    μ[0] = q.T@q

    for i in range(k-1):
        q__ = np.copy(q)
        q = 1 / δ[i] * (A@q - γ[i]*q - (δ[i-1] if i>0 else 0)*q_) 
        q_ = q__

        μ[i+1] = v.T@q

    return μ

def get_op(k,γ,δ):

    def pk(x,γ=γ,δ=δ):

        q = np.ones_like(x); q_ = np.zeros_like(x)
        for i in range(k):
            q__ = np.copy(q)
            q = 1 / δ[i] * (x*q - γ[i]*q - (δ[i-1] if i>0 else 0)*q_) 
            q_ = q__
                
        return q

    return pk
    
def get_op_expansion(c,γ,δ):

    def f(x,γ=γ,δ=δ):
        y = np.zeros_like(x)
        for k,ck in enumerate(c):
            pk =  get_op(k,γ,δ)
            y += ck*pk(x)

        return y

    return f
        
def get_chebT_recurrence(k,Emin,Emax):

    γ = (Emax+Emin)/2 * np.ones(k)
    δ = (Emax-Emin)/4 * np.ones(k)
    δ[0] *= np.sqrt(2)
    
    return (γ,δ)

def get_chebT_density(Emin,Emax):

    def σ(E):

        supp = (E>Emin)*(E<Emax)
        return (1/np.pi)/(np.sqrt((Emax-E)*(E-Emin)*supp)+1*(1-supp))*supp
        
    return σ

def get_connection_coeffs(α,β,γ,δ,fill=np.nan):
    
#    α,β = M
#    γ,δ = N
    
    km1,km2 = len(α),len(β);
    kn1,kn2 = len(γ),len(δ);
    
    assert (km1==km2 or km1==km2+1) and (kn1==kn2 or kn1==kn2+1),\
           "Jacobi matrices must be size k,k or k+1,k"
    
    C = np.full((min(km2,kn2)+1,\
                 min(km2,kn1+kn2)+1)\
                ,fill,dtype=np.longdouble)
    
    C[0,0] = 1
    for j in range(1,min(km2,kn1+kn2)+1):
        for i in range(0,min(j,kn1+kn2-j)+1):
            C[i,j] =( ( δ[i-1]*C[i-1,j-1]          if i-1>=0 else 0.)\
                      +( (γ[i]-α[j-1])*C[i,j-1]    if i<=j-1 else 0.)\
                      +( δ[i]*C[i+1,j-1]           if i+1<=j-1 else 0.)\
                      -( β[j-2]*C[i,j-2]           if i<=j-2 else 0.)\
                     )/β[j-1]
            
    return C

def jackson_weights(k):
    return (1/(k+1))*((k-np.arange(k)+1)*np.cos(np.pi*np.arange(k)/(k+1))+np.sin(np.pi*np.arange(k)/(k+1))/np.tan(np.pi/(k+1)))

def get_chebU_recurrence(k,Emin,Emax):

    γ = (Emax+Emin)/2 * np.ones(k)
    δ = (Emax-Emin)/4 * np.ones(k)
    
    return (γ,δ)

def get_chebU_density(Emin,Emax):

    def σ(E):

        supp = (E>Emin)*(E<Emax)
        return 8/(np.pi*(Emax-Emin)**2)*(np.sqrt((Emax-E)*(E-Emin)*supp))

    return σ


def get_gq(k,α,β):
    
    try:
        θ,S = sp.linalg.eigh_tridiagonal(α[:k],β[:k-1])
    except:
        T = np.diag(γ[:k]) + np.diag(β[:k-1],1) + np.diag(δ[:k-1],-1)
        θ,S = sp.linalg.eigh(T)

    ω = np.abs(S[0])**2
    
    return θ,ω
    
def get_multiint_op_recurrence(γδs,weights,k):
    
    l = len(weights)

    
    GQ_nodesweights = [get_gq(k,γ,δ) for (γ,δ) in γδs]

    GQ_nodes = np.array([gq[0] for gq in GQ_nodesweights]).flatten()
    GQ_weights = (np.array(weights)[:,None]*np.array([gq[1] for gq in GQ_nodesweights])).flatten()
    
    A = sp.sparse.spdiags(GQ_nodes,0,(k)*l,(k)*l)
    v = np.sqrt(GQ_weights)
    
    (γ1,δ1) = lanczos_reorth(A,v,k+1,reorth=True)
    
    return γ1[:k],δ1[:k]