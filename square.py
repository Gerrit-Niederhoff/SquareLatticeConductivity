import numpy as np
from numpy import sqrt, sin, cos, exp, log
from numpy import pi as π
def ξ_kinetic(kx,ky,μ=0,t=1,**kwargs):
    arr=-2*t*(cos(kx)+cos(ky))-μ
    return arr
def vel(kx,ky,t=1,qx=0,**kwargs):
    """
    Calculates the bare photon vertex of the kinetic hamiltonian,
    returns a (*k,x/y,2,2)-array
    """
    arr1 = 2*t*np.array([sin(kx+qx)+0*ky,sin(ky)+0*kx])
    arr2 = 2*t*np.array([sin(kx-qx)+0*ky,sin(ky)+0*kx])
    vert = np.array([
        [arr1,0*arr1],
        [0*arr1,arr2]
    ])
    return vert.transpose((3,2,0,1))
def HBdG(kx,ky,Δ,qx=0,**kwargs):
    ξ1 = ξ_kinetic(kx+qx,ky,**kwargs)
    ξ2 = -ξ_kinetic(-kx+qx,-ky,**kwargs)
    h = np.array([
        [ξ1,Δ],
        [Δ.conj(),ξ2]
    ]).swapaxes(1,-1).swapaxes(0,-2)
    return h
def φvector(kx,ky,choose=np.ones(5,dtype=bool)):
    """
    Returns a vector containing the 3 basis-functions for pairing: s,px,py.
    Shape will be (*k,dof)
    """
    p1 = (sin(kx)+sin(ky))/sqrt(2)    #Odd parity terms
    p2 = (sin(ky)-sin(kx))/sqrt(2)    #Odd parity terms
    d  = (cos(ky)-cos(kx))/sqrt(2)    #d-wave term
    s  = (cos(kx)+cos(ky))/sqrt(2)    #extended s-wave term
    onsite = np.ones_like(s)#Term due to onsite-interaction
    arr = np.moveaxis(np.array([s,d,p1,p2,onsite]),0,-1)
    return arr[...,choose]
def fermi(E,T=0,**kwargs):
    if T==0:
        f = np.zeros_like(E)
        f[E<0] = 1.0
    else:
        f = 1/(1+exp(E/T))
    return f
def findShell(ωc,N=100,**kwargs):
    k = np.linspace(-π,π,N,endpoint=False)
    kx,ky = np.meshgrid(k,k)
    E = ξ_kinetic(kx,ky,**kwargs)
    cond = abs(E)<ωc
    return kx[cond],ky[cond]
def Ytensor(kx,ky,**kwargs):
    φ = φvector(kx,ky,**kwargs)
    φ = np.concatenate((φ,1j*φ),axis=-1) #(*k,6)
    Y = np.array([
        [0*φ,       φ],
        [φ.conj(),0*φ]
    ])
    Y = np.transpose(Y,(2,3,0,1))
    return Y
### Functions for calculating the optical response without numerical diagonalization etc.
def v3(kx,ky,t=1,qx=0,qy=0,**kwargs):
    """
    Gives the τ_3 component of the x-component of the bare photon vertex, a (*k,x/y) array
    """
    x = t*(sin(kx+qx)-sin(kx-qx))
    y = t*(sin(ky+qy)-sin(ky-qy))
    arr = np.array([x,y])
    return arr.transpose((1,0))
def energy_terms(kx,ky,Δ,qx=0,qy=0,**kwargs):
    """
    Return ξ_bar, δ, E+ and E-
    """
    ξ1 = ξ_kinetic(kx+qx,ky+qy,**kwargs)
    ξ2 = ξ_kinetic(-kx+qx,-ky+qy,**kwargs)
    ξprime = 0.5*(ξ1-ξ2)
    ξbar = 0.5*(ξ1+ξ2)
    δ=sqrt(ξbar**2+abs(Δ)**2)
    Eplus = ξprime+δ
    Eminus = ξprime-δ
    return ξbar,δ,Eplus,Eminus
def bilayer_energy_terms(kx,ky,Δ,qx=0,qy=0,J=0,**kwargs):
    """
    Returns (k,2)-arrays for ξ_bar, δ, E+ and E-, where the second axis
    represents the lower and upper band due to the bilayer.
    """
    ξ1lower = ξ_kinetic(kx+qx,ky+qy,**kwargs)-J
    ξ2lower = ξ_kinetic(-kx+qx,-ky+qy,**kwargs)-J
    ξ1upper = ξ_kinetic(kx+qx,ky+qy,**kwargs)+J
    ξ2upper = ξ_kinetic(-kx+qx,-ky+qy,**kwargs)+J
    ξ1 = np.array([ξ1lower,ξ1upper]).transpose()
    ξ2 = np.array([ξ2lower,ξ2upper]).transpose()
    ξprime = 0.5*(ξ1-ξ2)
    ξbar = 0.5*(ξ1+ξ2)
    δ=sqrt(ξbar**2+abs(Δ)**2)
    Eplus = ξprime+δ
    Eminus = ξprime-δ
    return ξbar,δ,Eplus,Eminus