import rashba as rs
import numpy as np
CF = 1e-3
from IPython.display import clear_output
def couplingChunk(kx,ky,dA,Δ,ω,α=0.1,ωc=1,**kwargs):
    Yarray = rs.Ytensor(kx,ky,α=α)   #(Kx,Ky,14,4,4)
    H = rs.HBdG(kx,ky,Δ,α=α,**kwargs)#(Kx,Ky,4,4)
    E,V = np.linalg.eigh(H)     #(Kx,Ky,4) and (Kx,Ky,4,4), respectively
    Vdagger = V.conj().swapaxes(-1,-2)

    E_dif = E[:,:,:,None]-E[:,:,None,:] # (Kx,Ky,i,j)
    Fermi = rs.fermi(E)
    Fermi_dif = Fermi[:,:,:,None]-Fermi[:,:,None,:] #(K,i,j)
    #Actually the product of propagator and the fermi-difference
    propagator = Fermi_dif[:,:,None,:,:]/(ω[None,None,:,None,None]+E_dif[:,:,None,:,:]+1j*CF) #(Kx,Ky,w,i,j)

    Π = dA/2*np.einsum(
        'XYwij,XYpqij->wpq',
        propagator,Yarray[:,:,:,None,:,:]*(Yarray[:,:,None,:,:,:].swapaxes(-1,-2))
    )  
    return Π
def coupling(V_u,V_g,Δ,ω,N=100,ωc=1,α=0.1,**kwargs):
    dA = 1/N**2
    k = np.linspace(-np.pi,np.pi,N,endpoint=False)
    #Splitting into chunks:
    nk=N
    nω=len(ω)
    kchunks=1
    ωchunks=1
    #Divide array-length by 2 until it is below some managable limit (chosen as 200 because that seems to still work)
    while(nk>200):
        nk//=2
        kchunks*=2
    while(nω>200):
        nω//=2
        ωchunks*=2

    ω_intervals = np.split(ω,ωchunks)
    k_intervals = np.split(k,kchunks)

    #Splitting the gap-function along y-direction first
    Δchunks = np.array(np.split(Δ,kchunks,axis=-3))
    #And then along x:
    Δchunks = np.array(np.split(Δchunks,kchunks,axis=-4))
    print(np.shape(Δchunks))
    
    coupling = np.array([-V_g,-V_u,-V_u,-V_u,-V_u,-V_u,-V_u,-V_g,-V_u,-V_u,-V_u,-V_u,-V_u,-V_u])
    Π=[]
    for l,ω_interval in enumerate(ω_intervals):
        Π_sum = np.zeros((nω,14,14),dtype=complex)
        for i,kx_interval in enumerate(k_intervals):
            for j,ky_interval in enumerate(k_intervals):
                print(f"Frequency interval {l+1} of {ωchunks}")
                print(f"k_x interval {i+1} of {kchunks}")
                print(f"k_y interval {j+1} of {kchunks}")
                clear_output(wait=True)
                Π_chunk = couplingChunk(
                    kx_interval[:,None]+0*ky_interval[None,:],
                    ky_interval[None,:]+0*kx_interval[:,None],
                    Δchunks[i,j,...],
                    ω_interval,
                    dA,
                    ωc = ωc,
                    α=α,
                    **kwargs
                )
                Π_sum+=Π_chunk
        Π.append(Π_sum)
    #End of Frequency-loop:
    Π = np.array(Π).reshape(len(ω),14,14)
    Ueffinv = -np.diag(1/coupling)[None,...]+Π
    return Ueffinv