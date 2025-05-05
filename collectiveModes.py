import rashba as rs
import conductivity as cnd
import numpy as np
def minimalGap(kx,ky,η,**kwargs):
    ξ0 = rs.ξsquarelattice(kx,ky,**kwargs)
    γ = 0*np.linalg.norm(rs.γcoupling(kx,ky,**kwargs),axis=0)
    shift = np.array([γ,-γ])
    φ = rs.φsimplified(kx,ky)
    Δ = np.array([np.dot(φ,η), np.dot(φ.conj(),η)])#2,k
    δ = np.sqrt(ξ0**2+abs(Δ)**2)#2,k
    #Get full spectrum of energies:
    E = np.array([shift+δ,shift-δ]).reshape(4,len(kx))
    Enegative = E.copy()
    Enegative[E>0]*=-np.inf
    Epositive = E.copy()
    Epositive[E<0]*=-np.inf
    allDifferences = Epositive[:,None,:]-Enegative[None,:,:]
    return np.min(allDifferences)
def findMinima(arr,cut=np.inf):
    """
    Takes a positive 1-d numpy array and identifies local minima within, with first and last elements excluded.
    Returns the indices of the local minima within the array.
    """
    next = np.append(arr[1:],-1)  #Leaving out the first element
    prev = np.insert(arr[:-1],0,-1) #Leaving out the last element
    mincondition = (prev>arr) & (next>arr) &(arr<cut) #True for all indices where both the next and previous elements are larger
    ind = np.asarray(mincondition).nonzero()[0]
    return ind
def findModes(V,η,ωmax, Nk=400,Nω=800,ωc=1,ωmin=None,cut=0.05,nmodes=10,**kwargs):
    if ωmin==None:
        ωmin = ωmax/50
    ωarray = np.linspace(ωmin,ωmax,Nω)
    σ,VeffInv,Q,Qm = cnd.σ_simplified(V,η,ωarray,N=Nk,ωc=ωc,**kwargs)
    λ = abs(np.linalg.eigvals(VeffInv))
    index= findMinima(np.min(λ,axis=1),cut=cut)
    if len(index)>nmodes:
        return ωarray[index][:nmodes]
    else:
        missing = nmodes-len(index)
        extra = np.array([ωmax]*missing)
        frequencies = np.concatenate((ωarray[index],extra))
        return frequencies
def blockModes(V,η,m1,m2,ωmax,Nk=400,Nω=800,ωc=1,ωmin=None,**kwargs):
    """
    Same thing as findModes, but only for the two given modes.
    """
    if ωmin==None:
        ωmin = ωmax/50
    ωarray = np.linspace(ωmin,ωmax,Nω)
    σ,VeffInv,Q,Qm = cnd.σ_simplified(V,η,ωarray,N=Nk,ωc=ωc,**kwargs)
    block = VeffInv[:,
        [
            [m1,m1],
            [m2,m2]
        ],
        [
            [m1,m2],
            [m1,m2]
        ]
    ]
    λ = abs(np.linalg.eigvals(block))
    index= findMinima(np.min(λ,axis=1))
    if len(index)>0:
        return ωarray[index][0]
    else:
        return 0