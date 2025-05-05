import numpy as np
from IPython.display import clear_output
from numpy import sqrt, sin, cos, exp, log
from numpy import pi as π
import square as sq
import freeDispersion as fr
τminus = np.array([
    [0,0],
    [1,0]
])
def findGap(V=np.ones(5),η_in = np.ones(5),N=100,ωc=20,maxiter=100,tol=1e-5,quiet=False,**kwargs):
    kx,ky = sq.findShell(ωc,N,**kwargs)        
    dA = 1/N**2
    φ = sq.φvector(kx,ky)
    Δ = np.dot(φ,η_in)
    H = sq.HBdG(kx,ky,Δ,**kwargs)
    for i in range(maxiter):
        E,Umatrix = np.linalg.eigh(H) #(*k,2) and (*k,2,2)
        Udagger = Umatrix.conj().swapaxes(-1,-2)
        fermi = sq.fermi(E,**kwargs) #(*k,2)
        τnew = Udagger @ τminus[None,:,:] @ Umatrix     #(*k,2,2)
        η = -V*dA*np.einsum('Ka,Kj,Kjj->a',φ,fermi,τnew)
        Δnew = np.dot(φ,η)
        diff = np.max(abs(Δnew-Δ))
        if (not quiet): print(f"Maximal difference is {diff}")
        Δ=Δnew
        if diff<tol:
            print(f"Converged after {i+1} iterations")
            break
        elif i+1==maxiter:
            print(f"Didn't converge, returning last value.")
        else:
            H[:,0,1] = Δ
            H[:,1,0] = Δ.conj()
            if((i+1)%10==0 and (not quiet)):
                clear_output(wait=True)
    return η
def free_energy(η,V,N=100,**kwargs):
    η_nonzero = η[V!=0]
    V_nonzero = V[V!=0]
    mf_term = np.sum(η_nonzero.conj()*η_nonzero/V_nonzero)
    kx,ky = sq.findShell(100,N=N,**kwargs)
    nk = len(kx)
    φ = sq.φvector(kx,ky)
    Δ = np.dot(φ,η)
    ξbar,δ,Eplus,Eminus = sq.energy_terms(kx,ky,Δ,**kwargs)
    energyterm = np.sum(Eminus[Eminus<0])/N**2+np.sum(Eplus[Eplus<0])/N**2
    return mf_term.real+energyterm
def findGapFast(V=np.ones(5),η_in = np.ones(5),N=100,ωc=20,maxiter=100,tol=1e-5,quiet=False,**kwargs):
    kx,ky = sq.findShell(ωc,N,**kwargs)        
    dA = 1/N**2
    φ = sq.φvector(kx,ky)
    Δ = np.dot(φ,η_in)
    ξbar,δ,Eplus,Eminus = sq.energy_terms(kx,ky,Δ,**kwargs)
    print(η_in[[1,-1]])
    for i in range(maxiter):
        kern = Δ*(sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs))/(2*δ)
        η = -V*dA*np.einsum('Ka,K->a',φ,kern)
        Δnew = np.dot(φ,η)
        diff = np.max(abs(Δnew-Δ))
        if(not quiet):
            print(f"Maximal difference is {diff}")
            print(np.mean(abs(kern)))
            print(η[[1,-1]])
        Δ=Δnew
        δnew =np.sqrt(ξbar**2+abs(Δ)**2)
        Eplus+= -δ+δnew
        Eminus+= δ- δnew
        δ =δnew
        if diff<tol:
            print(f"Converged after {i+1} iterations")
            break
        elif i+1==maxiter:
            print(f"Didn't converge, returning last value.")
        else:
            if((i+1)%10==0 and (not quiet)): clear_output(wait=True)
    return η
def findGapVeryFast(Vs,Vd,η_s_in=1,η_d_in=1,N=100,ωc=20,maxiter=100,tol=1e-5,quiet=False,use="square",**kwargs):
    """
    ASSUMES that the only stable relative phase between s- and d is pi/2, works with real coefficients.
    Could produce incorrect results if the assumption is wrong.
    """
    if use=="square":
        file = sq
    elif use=="free":
        file = fr
    else:
        raise ValueError("Only able to use 'free' or 'square'.")
    dA = 1/N**2
    kx,ky = file.findShell(ωc,N,**kwargs)    
    print    
    φ = file.φvector(kx,ky,choose=np.array([False,True,False,False,True]))
    Δ = 1j*φ[:,0]*η_d_in+η_s_in
    ηs = η_s_in
    ηd = η_d_in
    ξbar,δ,Eplus,Eminus = file.energy_terms(kx,ky,Δ,**kwargs)
    for i in range(maxiter):
        kern_s = ηs*(file.fermi(Eplus,**kwargs)-file.fermi(Eminus,**kwargs))/(2*δ)
        kern_d = ηd*φ[:,0]*(file.fermi(Eplus,**kwargs)-file.fermi(Eminus,**kwargs))/(2*δ)
        ηs = -Vs*dA*(np.sum(kern_s)).real
        ηd = -Vd*dA*(np.sum(φ[:,0]*kern_d)).real
        Δnew = 1j*φ[:,0]*ηd+ηs
        diff = np.max(abs(Δnew-Δ))
        if(not quiet): print(f"Maximal difference is {diff}")
        Δ=Δnew
        δnew =np.sqrt(ξbar**2+abs(Δ)**2)
        Eplus+= -δ+δnew
        Eminus+= δ- δnew
        δ =δnew
        if diff<tol:
            print(f"Converged after {i+1} iterations")
            break
        elif i+1==maxiter:
            print(f"Didn't converge, returning last value.")
        else:
            if((i+1)%10==0 and (not quiet)):
                clear_output(wait=True)
    return ηs,ηd