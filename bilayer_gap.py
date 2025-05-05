import numpy as np
from IPython.display import clear_output
from numpy import sqrt, sin, cos, exp, log
from numpy import pi as π
import square as sq
def findGap(Vs,Vd,J,η_s_in=np.array([1,1]),η_d_in=np.array([1,1]),
            N=100,ωc=20,maxiter=100,tol=1e-5,quiet=False,**kwargs):
    """
    ASSUMES that the only stable relative phase between s- and d is pi/2, works with real coefficients.
    Could produce incorrect results if the assumption is wrong. Arguments are:
    Vs, Vd: 2x2 Matrices
    J: float 
    η_s_in,η_d_in (optional): 2-vectors
    """
    dA = 1/N**2
    kx,ky = sq.findShell(ωc,N,**kwargs)    
    φd = sq.φvector(kx,ky,choose=np.array([False,True,False,False,False]))[:,0]
    Δ = 1j*φd[:,None]*η_d_in[None,:]+η_s_in[None,:]
    ηs = η_s_in
    ηd = η_d_in
    ξbar,δ,Eplus,Eminus = sq.bilayer_energy_terms(kx,ky,Δ,J=J,**kwargs) #all (k,2)-arrays
    for i in range(maxiter):
        kern_s = (sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs))/(2*δ)
        kern_d = (φd[:,None])**2*(sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs))/(2*δ)
        ηs = -Vs@ ηs*dA*(np.sum(kern_s,axis=0))
        ηd = -Vd@ ηd*dA*(np.sum(kern_d,axis=0))
        Δnew = 1j*φd[:,None]*ηd[None,:]+ηs[None,:]
        diff = np.max(abs(Δnew-Δ))
        if(not quiet): print(f"Maximal difference is {diff}")
        Δ=Δnew
        δnew =np.sqrt(ξbar**2+abs(Δ)**2)
        Eplus +=-δ + δnew
        Eminus+= δ - δnew
        δ = δnew
        if np.max(abs(diff))<tol:
            print(f"Converged after {i+1} iterations")
            break
        elif i+1==maxiter:
            print(f"Didn't converge, returning last value.")
        else:
            if((i+1)%10==0 and (not quiet)):
                clear_output(wait=True)
    return ηs,ηd
def findGapComplex(Vs,Vd,J,η_s_in=np.ones(2),η_d_in=1j*np.ones(2),
            N=100,ωc=20,maxiter=100,tol=1e-5,quiet=False,**kwargs):
    """
    Doesn't assume an s+id form of the order parameter, works with general complex coefficients instead.
    """
    dA = 1/N**2
    kx,ky = sq.findShell(ωc,N,**kwargs)    
    φ = sq.φvector(kx,ky,choose=np.array([False,True,False,False,True]))#(k,μ)
    η = np.array([η_d_in,η_s_in]) #(μ,α)
    Δ = φ@η     #(k,μ)@(μ,α) --> (k,α)
    ξbar,δ,Eplus,Eminus = sq.bilayer_energy_terms(kx,ky,Δ,J=J,**kwargs) #all (k,α)-arrays
    V = np.array([Vd,Vs]) #(μ,α,β)
    for i in range(maxiter): 
        kern = Δ*(sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs))/(2*δ) #(k,α)
        ηnew = -dA*np.tensordot(kern,φ,axes=((0,),(0,))) # α,μ: intermediate result
        ηnew = np.einsum('mab,bm->ma',V,ηnew)
        #ηnew = -dA*np.einsum('mab,km,kb->ma',V,φ,kern) 
        diff =ηnew-η
        if(not quiet):
            print(f"Maximal difference is {np.max(abs(diff))}, max gap is {ηnew[1,0]}")
            #print(np.mean(abs(kern),axis=0))
            #print(η[:,1])
        η+=(diff)
        Δ = φ@η
        δnew =np.sqrt(ξbar**2+abs(Δ)**2)
        Eplus +=-δ + δnew
        Eminus+= δ - δnew
        δ = δnew
        if np.max(abs(diff))<tol:
            print(f"Converged after {i+1} iterations")
            break
        elif i+1==maxiter:
            print(f"Didn't converge, returning last value.")
        else:
            if((i+1)%10==0 and (not quiet)):
                clear_output(wait=True)
    return η
def free_energy(Vs,Vd,J,ηs,ηd,N=400,ωc=20,**kwargs):
    kx,ky = sq.findShell(ωc,N=N,**kwargs)
    Vsinv = np.linalg.inv(-Vs)
    Vdinv = np.linalg.inv(-Vd)
    mfs = -(ηs.conj()@Vsinv@ηs).real
    mfd = -(ηd.conj()@Vdinv@ηd).real

    φd = sq.φvector(kx,ky)[:,1]
    ξbar,δ,Eplus,Eminus = sq.bilayer_energy_terms(kx,ky,ηs[None,:]+ηd[None,:]*φd[:,None],**kwargs)
    energyterm = np.sum(Eminus[Eminus<0])/N**2+np.sum(Eplus[Eplus<0])/N**2
    return mfs+mfd+energyterm
