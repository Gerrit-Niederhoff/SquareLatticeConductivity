import conductivity as cnd
import numpy as np
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
def subGapModes(VeffInv,ω,cut=np.inf):
    """
    Find all modes below the minimal gap. Takes a (ω,8,8)- array of inverse effective couplings,
    computes the local minima of the lowest eigenvalue and returns an array of frequencies where
    these local minima occur.
    """
    eig = abs(np.linalg.eigvals(VeffInv))
    minEig = np.min(eig,axis=1)#Take the minimal eigenvalues AT EACH FREQUENCY
    modes = findMinima(minEig,cut)
    return ω[modes]
def swaveModes(Vs,Vd,ηs,ηd,**kwargs):
    ω=np.linspace(1e-4,2.1*np.max(abs(ηs)),800)
    #Largest sub-gap index:
    minGapInd = np.asarray(ω<=2*np.min(abs(ηs))).nonzero()[0][-1]
    bare,coll,VeffInv,Q,Qm = cnd.σ_bilayer(Vs,Vd,ηs+0j,ηd,ω,**kwargs)
    σ = bare[:,0,0]+coll[:,0,0]
    modes = subGapModes(VeffInv[:minGapInd,...],ω[:minGapInd])
    if len(modes)==1:
        bs1 = np.argmax(σ[minGapInd-20:minGapInd+20])+minGapInd-20
        bs2 = np.argmax(σ[minGapInd+20:])+minGapInd+20
        modes = np.append(modes,[ω[bs1],ω[bs2]])
    elif len(modes)==2:
        bs2 = np.argmax(σ[minGapInd:])+minGapInd
        modes = np.append(modes,ω[bs2])
    return modes
def sidModes(Vs,Vd,ηs,ηd,**kwargs):
    ω=np.linspace(0.2*np.min(abs(ηs)),2*np.max(abs(ηs)),200)
    bare,coll,VeffInv,Q,Qm = cnd.σ_bilayer(Vs,Vd,ηs+0j,ηd,ω,**kwargs)
    modes = subGapModes(VeffInv,ω)
    return modes[:3]
def symmetricModes(Vs,Vd,ηs,ηd,**kwargs):
    ωmax = 2.01*abs(ηs[0])
    ω=np.linspace(ωmax/8,ωmax,600)
    bare,coll,VeffInv,Q,Qm = cnd.σ_bilayer(Vs,Vd,ηs,ηd,ω,**kwargs)
    modes = subGapModes(VeffInv,ω)
    return modes[:3]
    
