import numpy as np
from numpy import sqrt, sin, cos, exp
π = np.pi
def γcoupling(kx,ky,α=.1,**kwargs):
    """
    The Rashba-SOC term
    """
    return α*np.array([-sin(ky)+0*kx,sin(kx)+0*ky])
def tλ(kx,ky):
    """
    Calculate the phase-factor which distinguishes the gap from the rescaled gap.
    returns a (2,2,*k)-array
    """
    phase = (-sin(ky)-1j*sin(kx))/sqrt(sin(kx)**2+sin(ky)**2)
    return np.array([
        [phase,-phase],
        [phase,-phase]
    ])

def ξsquarelattice(kx,ky,t=1,μ=0,**kwargs):
    """
    The basic dispersion on the square lattice, without SOC.
    """
    return -2*t*(cos(kx)+cos(ky))-μ

def ξ_diag(kx,ky,λ,t=1,α=0.1,μ=0,**kwargs):
    """
    Returns one of the Rashba-bands as specified by λ=+-1
    Shape of return value will be the same as kx and ky.
    """
    return ξsquarelattice(kx,ky,t=t)+λ*np.linalg.norm(γcoupling(kx,ky,α=α),axis=0)-μ
def vel(kvec,t=1,α=0.1,**kwargs):
    """
    kvec: numpy array (2,*kdims)
    returns a (2,2,2,*kdims)-array, first axis is x/y-direction
    """
    n = np.linalg.norm(sin(kvec),axis=0)
    term1 = 2*t*sin(kvec)
    term2 = α*cos(kvec)*sin(kvec)/n
    o = np.zeros_like(term1)
    return np.array([
        [term1+term2,o,o,o],
        [o,term1-term2,o,o],
        [o,o,term1+term2,o],
        [o,o,o,term1-term2]
    ]).swapaxes(1,2).swapaxes(0,1)
def HBdG(kx,ky,Δ,**kwargs):
    """
    Takes arrays kx,ky need to have the same shape *k (1D or 2D)
    and  Δ with shape (*k,2,2)
    Returns a (*k,4,4)-array, the BdG-Hamiltonian evaluated at each momentum (kx,ky),
    with the gap being the corresponding entry of Delta.
    """
    ξplus = ξ_diag(kx,ky,1,**kwargs)
    ξminus = ξ_diag(kx,ky,-1,**kwargs)
    o = np.zeros_like(ξplus)
    H=np.array([
        [ξplus,o,o,o],
        [o,ξminus,o,o],
        [o,o,-ξplus,o],
        [o,o,o,-ξminus]
    ],dtype=complex).swapaxes(1,-1).swapaxes(0,-2)
    H[...,:2,2:] = Δ
    H[...,2:,:2] = Δ.conj().swapaxes(-2,-1)
    return H
def fermi(E,T=0,**kwargs):
    """
    Returns an array of fermi-distribution-values for an array of energies.
    """
    if T==0:
        cond = E>0
        f = np.zeros_like(E)
        f[cond]=0
        f[~cond]=1
    else:
        f = 1/(exp(E/T)+1)
    return f
def fermi_surface(E,tol=0.01):
    return abs(E)<tol
def findShell(N=100,ωc=10,**kwargs):
    """
    Calculates the kx and ky values in an energy shell of width ωc
    around the fermi-surface, of the basic hamiltonian (without SOC). Assuming small
    SOC, this shell can be used for the superconducting pairing.
    Also for sufficiently large ωc it makes no difference.
    """
    k = np.linspace(-π,π,N,endpoint=False)
    kx,ky = np.meshgrid(k,k)
    E = ξsquarelattice(kx,ky,**kwargs)
    cond = abs(E)<ωc
    return kx[cond],ky[cond]
def findShells(N=100,ωc=1,**kwargs):
    """
    def findShells(N=100,ωc=1,**kwargs):
        return kx_FS,ky_FS,plusShell,minusShell,total
    Returns list of kx and ky-values which are in a shell around either fermi-surface,
    boolean lists of the same length describing whether a given point is in the shell around either band,
    and a boolean grid describing where the shells lie in the total grid
    """
    k = np.linspace(-π,π,N,endpoint=False)
    plusBand = ξ_diag(k[:,None],k[None,:],λ=1,**kwargs)
    minusBand = ξ_diag(k[:,None],k[None,:],λ=-1,**kwargs)
    condplus = fermi_surface(plusBand,tol=ωc)
    condminus = fermi_surface(minusBand,tol=ωc)
    total = condplus|condminus
    kx_FS = (k[:,None]+0*k[None,:])[total]
    ky_FS = (0*k[:,None]+k[None,:])[total]
    plusShell=condplus[total]
    minusShell=condminus[total]
    return kx_FS,ky_FS,plusShell,minusShell,total
σx = np.array([
    [0,1],
    [1,0]
])
σy = np.array([
    [0,-1j],
    [1j,0]
])
mplus = np.array([
    [1,0],
    [0,0]
])
mminus = np.array([
    [0,0],
    [0,1]
])
def φvector(kx,ky,**kwargs):
    """
    Returns a vector filled with the φ-matrices used as a basis.
    Shape is (nbasis,2,2,*kdims).
    """
    γ1, γ2 = -sin(ky),sin(kx)
    norm = sqrt(γ1**2+γ2**2)
    #γ1/=norm
    #γ2/=norm
    o=0*γ1
    φ0 = np.array([
        [1+o,o],
        [o,1+o]
    ])/sqrt(2)
    φ1 = np.array([
        [γ1**2,-1j*γ1*γ2],
        [1j*γ1*γ2,-γ1**2]
    ])
    φ2 = np.array([
        [γ1*γ2,1j*γ1**2],
        [-1j*γ1**2,-γ1*γ2]
    ])
    φ3 = np.array([
        [o,γ1],
        [γ1,o]
    ])
    φ4 = np.array([
        [γ1*γ2,-1j*γ2**2],
        [1j*γ2**2,-γ1*γ2]
    ])
    φ5 = np.array([
        [γ2**2,1j*γ1*γ2],
        [-1j*γ1*γ2,-γ2**2]
    ])
    φ6 = np.array([
        [o,γ2],
        [γ2,o]
    ])
    φ = np.array([φ0,φ1,φ2,φ3,φ4,φ5,φ6])
    φalt = np.array([
        φ0,
        (φ2-φ4)/sqrt(2),
        (φ2+φ4)/sqrt(2),
        (φ1+φ5)/sqrt(2),
        (φ1-φ5)/sqrt(2),
        φ3,
        φ6
    ])
    A1 =1+o
    B1 = γ1**2-γ2**2
    B2 = γ1*γ2
    kshape = B1.shape
    bands = np.array([mplus,mminus,-σy/sqrt(2)]*3+[σx]*2)   #(11,2,2)
    irreps = np.array([A1]*3+[B1]*3+[B2]*3+[γ1,γ2]) #(11,*k)
    φirreps = bands[...,*((None,)*len(kshape))]*irreps[:,None,None,...]
    return φirreps
def Ytensor(kx,ky,nonzeroPairing=np.ones(7,dtype=bool),**kwargs):
    """
    Calculates the full fluctuation basis-array. Axis [-2] enumerates the components of the fluctuation.
    The first half of that axis corresponds to real, the second half to imaginary fluctuations (meaning these become 
    multiplied by i)
    Takes a kx and ky-array (same shape, 1D or 2D), return a (*K,2*nbasis,4,4)-array.
    """
    #Get the basis-functions
    kdims = len(np.shape(kx))
    nc = len(nonzeroPairing[nonzeroPairing])
    φ = φvector(kx,ky,**kwargs)
    B = np.zeros_like(φ,dtype=bool) | nonzeroPairing[(...,None,None)+(None,)*kdims]
    φ = φ[B].reshape(nc,2,2,*(kx.shape))
    #make a twice-as-long vector, for the real and imaginary part
    φxt=np.concatenate((φ,1j*φ),axis=0) #(2*nc,2,2,*kdims)

    #take the hermitian conjugate of each 2x2-matrix
    zero = np.zeros_like(φxt)
    arr = np.array([
        [zero,                     φxt],
        [φxt.conj().swapaxes(1,2),zero]
    ])#(2,2,14,2,2,*kdims)
    k_axes = [5+i for i in range(kdims)]
    #First move the nc-axis to the first position,
    #then swap the 2nd and 3rd of the 2-dimensional axes, to get the correct blockmatrix
    arr = arr.transpose(*k_axes,2,0,3,1,4)
    return arr.reshape(*(kx.shape),2*nc,4,4)
def φsimplified(kx,ky):
    """
    Return the basis-functions for a1,e1,e2 with only interband-pairing.
    Returns a (*k,3)-array
    """
    a1 = 1j*np.ones_like(kx)/sqrt(2)
    e1 = sin(kx)
    e2 = sin(ky)
    norm = 1#np.sqrt(e1**2+e2**2)+1e-20
    return np.array([a1,e1/norm,e2/norm]).transpose()
def v3(kx,ky,α=0.1,**kwargs):
    """
    Returns the derivative of |γ_k| as a (k,2)-array
    """
    
    norm = sqrt(sin(kx)**2+sin(ky)**2)+1e-20
    vec = α/norm*np.array([
        sin(kx)*cos(kx),
        sin(ky)*cos(ky)
    ])
    return vec.transpose()