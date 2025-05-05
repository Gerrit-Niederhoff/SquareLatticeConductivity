import rashba as rs
import numpy as np
import math
from time import time as t
from IPython.display import clear_output
import math as m
CF = 1e-10

def calculate_σ(V,Δ,ω,N=100,ωc=1,returnUinv=False,fullBZ=True,**kwargs):
    kx_FS,ky_FS,plusShell,minusShell,mask = rs.findShells(N=N,ωc=ωc,**kwargs)
    nω=len(ω)
    ωchunks=1
    while(nω>200):
        nω//=2
        ωchunks*=2
    ω_intervals = np.split(ω,ωchunks)
    #Only use components which have non-zero interaction strength
    nonzero_coupling = (V!=0)
    coupling = -V[nonzero_coupling]
    coupling = np.concatenate((coupling,coupling))
    nc = len(coupling)#Number of considered components
    σ_quasiparticles=[]
    Π=[]
    Q=[]
    Qminus = []
    if fullBZ:  #sum over 2D grid, covering the entire BZ
        #Create (N,N) Boolean Arrays describing the pairing shells
        plusShellGrid = np.zeros_like(mask,dtype=bool)
        plusShellGrid[mask]=plusShell
        minusShellGrid = np.zeros_like(mask,dtype=bool)
        minusShellGrid[mask] = minusShell
        sharedShellGrid = minusShellGrid & plusShellGrid

        #A boolean filter-array indicating allowed pairings: 
        allowedPairing = np.zeros((N,N,nc,4,4),dtype=bool)
        allowedPairing[:,:,:,[0,2],[2,0]]=plusShellGrid[:,:,None,None]
        allowedPairing[:,:,:,[1,3],[3,1]]=minusShellGrid[:,:,None,None]
        allowedPairing[:,:,:,[0,1,2,3],[3,2,1,0]]=sharedShellGrid[:,:,None,None]
        dA = 1/N**2
        k = np.linspace(-np.pi,np.pi,N,endpoint=False)
        #Splitting into chunks:
        nk=N
        kchunks=1
        #Divide array-length by 2 until it is below some managable limit (chosen as 200 because that seems to still work)
        while(nk>200):
            nk//=2
            kchunks*=2
        
        k_intervals = np.split(k,kchunks)

        #Splitting the gap-function along y-direction first
        Δchunks = np.array(np.split(Δ,kchunks,axis=-3))
        #And then along x:
        Δchunks = np.array(np.split(Δchunks,kchunks,axis=-4))
        print(np.shape(Δchunks))
        allowedPairingchunks = np.array(np.split(allowedPairing,kchunks,axis=-4))
        allowedPairingchunks = np.array(np.split(allowedPairingchunks,kchunks,axis=-5))
       
        for l,ω_interval in enumerate(ω_intervals):
            σ_QP_sum = np.zeros((nω,2,2))
            Π_sum = np.zeros((nω,nc,nc),dtype=complex)
            Q_sum = np.zeros((nω,2,nc),dtype=complex)
            Qm_sum = np.zeros((nω,2,nc),dtype=complex)
            for i,kx_interval in enumerate(k_intervals):
                for j,ky_interval in enumerate(k_intervals):
                    print(f"Frequency interval {l+1} of {ωchunks}")
                    print(f"k_x interval {i+1} of {kchunks}")
                    print(f"k_y interval {j+1} of {kchunks}")
                    clear_output(wait=True)
                    σ_QP_chunk,Π_chunk,Q_chunk,Qm_chunk = calculate_chunk(
                        kx_interval[:,None]+0*ky_interval[None,:],
                        ky_interval[None,:]+0*kx_interval[:,None],
                        Δchunks[i,j,...],
                        ω_interval,
                        dA,
                        allowedPairingchunks[i,j,...],
                        nonzeroPairing=nonzero_coupling,
                        **kwargs
                    )
                    σ_QP_sum+=σ_QP_chunk
                    Π_sum+=Π_chunk
                    Q_sum+=Q_chunk
                    Qm_sum+=Qm_chunk
            σ_quasiparticles.append(σ_QP_sum)
            Π.append(Π_sum)
            Q.append(Q_sum)        
            Qminus.append(Qm_sum)
        #End of Frequency-loop for the Full BZ case
    else:   #Just use energy-shells around the Fermi surface

        #Calculating the relevant momenta around the fermi-surface
        Δshell = Δ[mask]
        nk = len(kx_FS)
        sharedShell = plusShell & minusShell
        allowedPairing = np.zeros((nk,nc,4,4),dtype=bool)
        allowedPairing[:,:,[0,2],[2,0]]=plusShell[:,None,None]
        allowedPairing[:,:,[1,3],[3,1]]=minusShell[:,None,None]
        allowedPairing[:,:,[0,1,2,3],[3,2,1,0]]=sharedShell[:,None,None]
        kchunks = math.ceil(nk/40000)# Split into intervals that are at most 40000 elements long
        dA = 1/N**2
        kx_intervals = np.array_split(kx_FS,kchunks)
        ky_intervals = np.array_split(ky_FS,kchunks)
        Δ_intervals = np.array_split(Δshell,kchunks)
        pairing_intervals = np.array_split(allowedPairing,kchunks)
        for i,ω_interval in enumerate(ω_intervals):
            σ_QP_sum = np.zeros((nω,2,2))
            Π_sum = np.zeros((nω,nc,nc),dtype=complex)
            Q_sum = np.zeros((nω,2,nc),dtype=complex)
            Qm_sum = np.zeros((nω,2,nc),dtype=complex)
            for j in range(kchunks):
                print(f"Frequency interval {i+1} of {ωchunks}")
                print(f"Momentum interval {j+1} of {kchunks}")
                clear_output(wait=True)
                σ_QP_chunk,Π_chunk,Q_chunk,Qm_chunk = calculate_chunk(
                    kx_intervals[j],
                    ky_intervals[j],
                    Δ_intervals[j],
                    ω_interval,
                    dA,
                    pairing_intervals[j],
                    nonzeroPairing=nonzero_coupling,
                    **kwargs
                )
                σ_QP_sum+=σ_QP_chunk
                Π_sum+=Π_chunk
                Q_sum+=Q_chunk
                Qm_sum+=Qm_chunk
            σ_quasiparticles.append(σ_QP_sum)
            Π.append(Π_sum)
            Q.append(Q_sum)        
            Qminus.append(Qm_sum)
    σ_quasiparticles = np.array(σ_quasiparticles).reshape(len(ω),2,2)
    Π = np.array(Π).reshape(len(ω),nc,nc)
    Q =np.array(Q).reshape(len(ω),2,nc)
    Qminus = np.array(Qminus).reshape(len(ω),2,nc)
    Ueffinv = np.diag(1/coupling)[None,...]-Π
    Ueff = np.linalg.inv(Ueffinv)
    σ_collective = np.einsum(
        'wap,wpq,wbq->wab',
        Q,Ueff,Qminus
    )
    σ_collective = (0.5j*σ_collective/ω[:,None,None]).real
    if returnUinv:
        return σ_quasiparticles,σ_collective,Ueffinv,Q,Qminus
    else:
        return σ_quasiparticles,σ_collective
def calculate_chunk(kx,ky,Δ,ω,dA,allowedPairing,nonzeroPairing,**kwargs):
    #Always use ... for the momentum axes, which could be 1 or 2
    kdimension = len(np.shape(kx))
    if kdimension==1:
        kstring = "K"
        newk_axes = (None,)
    else:
        kstring="XY"
        newk_axes = (None,None)
    #Basic ingredients:
    k_axes = [3+i for i in range(kdimension)]
    velocity = np.transpose(
        rs.vel(np.array([kx,ky]),**kwargs),
        (*k_axes,0,1,2)) #(*Kdims,x/y,4,4)
    Yarray = rs.Ytensor(kx,ky,nonzeroPairing=nonzeroPairing,**kwargs)   #(*Kdims,nc,4,4)
    Yarray[~allowedPairing]=0   #Turn of fluctuations in the forbidden pairing range

    H = rs.HBdG(kx,ky,Δ,**kwargs)#(*Kdims,4,4)
    E,V = np.linalg.eigh(H)     #(*Kdims,4) and (*Kdims,4,4), respectively
    Vdagger = V.conj().swapaxes(-1,-2)

    #Basic quantities
    E_dif = E[...,:,None]-E[...,None,:] # (*Kdims,i,j)
    Fermi = rs.fermi(E,**kwargs)
    Fermi_dif = Fermi[...,:,None]-Fermi[...,None,:] #(*kdims,i,j)
    #Actually the product of propagator and the fermi-difference
    propagator = Fermi_dif[...,None,:,:]/(ω[*newk_axes,:,None,None]+E_dif[...,None,:,:]+1j*CF) #(*kdimgs,w,i,j)
    propagator_neg = Fermi_dif[...,None,:,:]/(-ω[*newk_axes,:,None,None]+E_dif[...,None,:,:]+1j*CF) #(*kdims,w,i,j)
    #Transform the basis-function-tensor into the diagonal basis
    Yarray = (Vdagger[...,None,:,:]) @ Yarray @ (V[...,None,:,:]) #(*k,14,4,4)
    #Transform the velocity-array
    velocity = (Vdagger[...,None,:,:]) @ velocity @ (V[...,None,:,:])#(*k,x/y,4,4)


    #Summing over momentum and other d.o.f's
    σ_quasiparticles = dA*np.einsum(
        f"{kstring}wij,{kstring}aij,{kstring}bji->wab",
        propagator,velocity,velocity
    )
    σ_quasiparticles = (1j*σ_quasiparticles/(ω[:,None,None])).real
    Π = dA/2*np.einsum(
        f'{kstring}wij,{kstring}pqij->wpq',
        propagator,Yarray[...,:,None,:,:]*(Yarray[...,None,:,:,:].swapaxes(-1,-2))
    )    
    Q = dA*np.einsum(
        f'{kstring}wij,{kstring}apij->wap',
        propagator,
        velocity[...,:,None,:,:]*(Yarray[...,None,:,:,:].swapaxes(-1,-2))
    )
    Qminus = dA*np.einsum(
        f'{kstring}wij,{kstring}apij->wap',
        propagator_neg,
        velocity[...,:,None,:,:]*(Yarray[...,None,:,:,:].swapaxes(-1,-2))
    )
    return σ_quasiparticles,Π,Q,Qminus
def collective_σ(Ueffinv,Q,Qminus,ω):
    Ueff = np.linalg.inv(Ueffinv)
    σ_collective = np.einsum(
        'wap,wpq,wbq->wab',
        Q,Ueff,Qminus
    )
    σ_collective=(0.5j*σ_collective/ω[:,None,None]).real
    return σ_collective
def σ_simplified(V_in,η,ω,ωc=10,**kwargs):
    """
    Conductivity for a rashba-system with ONLY interband pairing, including s- and p-wave contributions.
    """
    #Multiply V with -1 to get the attractive potential.
    V = -V_in
    kx,ky = rs.findShell(ωc=ωc,**kwargs)
    #Calculate all polarization bubbles of the outer and inner block of the Green's function
    Φout,Πout,Qout,Qmout =simpleBubbles(η,ω,sign=1,ωc=ωc, **kwargs)
    Φin,Πin,Qin,Qmin =simpleBubbles(η,ω,sign=-1,ωc=ωc,**kwargs)
    Φ = Φin  + Φout
    Π = Πin  + Πout
    Q = Qin  + Qout
    Qm= Qmin + Qmout
    σQP = -0.5*(Φ/ω[:,None,None]).imag
    VeffInv = np.diag(
        np.concatenate((1/V,1/V))
    )[None,:,:]-Π/2
    Veff = np.linalg.inv(VeffInv)
    σ_collective = np.einsum('wap,wpq,wbq->wab',Qm,Veff,Q)
    σ_collective = -0.25*(σ_collective/ω[:,None,None]).imag
    return σQP+σ_collective,VeffInv,Q,Qm
def simpleBubbles(η,ω,N=400,reg=1e-4j,**kwargs):
    ωreg = ω+reg
    kx,ky = rs.findShell(N=N,**kwargs)
    nk = len(kx)
    nω = len(ω)
    kchunks = m.ceil(nk/5e4)
    #I'm still unsure about if this should be 1/N^2 or 1/nk:
    dA = 1/nk
    #Splitting kx,ky and Delta into smaller chunks to save memory
    kx_intervals = np.array_split(kx,kchunks)
    ky_intervals = np.array_split(ky,kchunks)
    integrals = np.zeros((5,6,nω),dtype=complex)
    vectorIntegrals = np.zeros((4,3,nω,2),dtype=complex)
    Φintegral = np.zeros((nω,2,2),dtype=complex)
    #ζ = np.zeros((nω,2),dtype=complex)
    #otherIntegrals = np.zeros((4,3,nω),dtype=complex)
    for j in range(kchunks):
        print(f"Momentum interval {j+1} of {kchunks}")
        #clear_output(wait=True)
        Ichunk,vIchunk,Φchunk=simplechunks(
              kx_intervals[j],ky_intervals[j],η,ωreg,dA,**kwargs
         )
        integrals+=Ichunk
        vectorIntegrals+=vIchunk
        Φintegral+=Φchunk
        #ζ += ζchunk
        #otherIntegrals += oIchunk
    Φintegral*=-4
    #ζ*=-4
    Iξξ,Iξ,Iii,Iri,Irr = integrals
    #Ii,Ir,Irξ,Iiξ = otherIntegrals
    QR,QI,QRξ,QIξ = vectorIntegrals
    
    #Creating the ν-indices of Π first (to later become the outer indices of the 4x4 matrix)
    Π = np.array([
         [-4*Iξξ-4*Iii,2j*ωreg*Iξ-4*Iri],
         [-2j*ωreg*Iξ-4*Iri,-4*Iξξ-4*Irr]
    ]).transpose((2,0,1,3))#(6,ν,ν',w)
    
    #The first axis contains the combinations s^2,sp_x,sp_y,p_x^2,p_xp_y,p_y^2
    #Arrange these combinations appropriately into the μ,μ'-indices.
    #Then rearrange: Frequency to the front, then ν,μ,ν',μ', so that reshape fuses the appropriate axes
    Π = np.array([
         [Π[0],Π[1],Π[2]],
         [Π[1],Π[3],Π[4]],
         [Π[2],Π[4],Π[5]]
    ]).transpose((4,2,0,3,1)).reshape(nω,6,6)
    #Similarly for Q. Here however, the component-axis contains only s,px,py. This means that one can reshape right-away
    Q = np.array([
         4*QRξ-2j*ωreg[None,:,None]*QI,
         -4*QIξ-2j*ωreg[None,:,None]*QR
    ]).transpose((2,3,0,1)).reshape(nω,2,6)
    Qm = np.array([
         4*QRξ+2j*ωreg[None,:,None]*QI,
         -4*QIξ+2j*ωreg[None,:,None]*QR
    ]).transpose((2,3,0,1)).reshape(nω,2,6)
    #Qm=Q.conj()
    ###Renormalize the bubbles due to the phase-integral:

    ##-4 I[|Δ|^2] becomes 8, because the [0] component contains a factor 1/2
    #kap = -8*(Irr[0]+Iii[0])
    ###Again, move frequency to the front, then nu,mu and reshape 
    #Λp = np.array([
    #    4*Irξ+2j*ωreg*Ii,
    #    -4*Iiξ+2j*ωreg*Ir
    #]).transpose((2,0,1)).reshape(nω,6)
    #Λm = np.array([
    #    4*Irξ-2j*ωreg*Ii,
    #    -4*Iiξ-2j*ωreg*Ir
    #]).transpose((2,0,1)).reshape(nω,6)
    #Λm = Λp.conj()
    ##Now renormalize:
    #print("Renormalizing, by: ")
    #print(np.mean(abs(Λp[:,:,None]*Λm[:,None,:]/kap[:,None,None])))
    #print(np.mean(abs(ζ[:,:,None]*Λm[:,None,:]/kap[:,None,None])))
    #print(np.mean(abs(Qm- ζ[:,:,None]*Λp[:,None,:]/kap[:,None,None])))
    #print(np.mean(abs(ζ[:,:,None]*ζ[:,None,:]/kap[:,None,None])))
    #Π = Π - Λp[:,:,None]*Λm[:,None,:]/kap[:,None,None]
    #Q = Q - ζ[:,:,None]*Λm[:,None,:]/kap[:,None,None]
    #Qm= Qm- ζ[:,:,None]*Λp[:,None,:]/kap[:,None,None]
    #Φintegral = Φintegral - ζ[:,:,None]*ζ[:,None,:]/kap[:,None,None]
    return Φintegral,Π,Q,Qm
def simplechunks(kx,ky,η,ω,dA,sign=1,q=np.zeros(2),**kwargs):
    """
    sign should be +1 for the outer block and -1 for the inner block
    """
    v3_0 = sign*rs.v3(kx,ky,**kwargs)
    v3 =v3_0 + 2*np.array([q[0]*np.cos(kx),q[1]*np.cos(ky)]).transpose()
    φs,φpx,φpy = rs.φsimplified(kx,ky).transpose()
    φs = sign*φs.imag
    

    #Build the gap. The s-wave component has the opposite sign in the inner block,
    #as required by symmetry
    Δ = 1j*φs* η[0]+η[1]*φpx+η[2]*φpy
    ξ0 = rs.ξsquarelattice(kx,ky,**kwargs)+np.dot(v3_0,q)
    δ = np.sqrt(ξ0**2+abs(Δ)**2)[:,None]
    #φs *= np.sqrt(renormalization[0])
    #φpx*= np.sqrt(renormalization[1])
    #φpy*= np.sqrt(renormalization[2])
    combi = np.array([φs**2,φpx*φs,φpy*φs,φpx**2,φpx*φpy,φpy**2])
    #The inner block energies are shifted downwards by γ instead of upwards:
    γ = sign*np.linalg.norm(rs.γcoupling(kx,ky,**kwargs),axis=0)
    γ= (γ+ 2*(q[0]*np.sin(kx)+q[1]*np.sin(ky)))[:,None]
    #Build integration kernels for the outer and inner block:
    kernel = ((rs.fermi(δ+γ)-rs.fermi(-δ+γ))/δ)/((ω[None,:])**2-4*(δ)**2)
    #Both are (k,ω)-arrays

    #Calculate all sorts of integrals:
    #The first axis in all of them represents all required combinations of basisfunctions,
    #for which that integral needs to be calculated
    
    ξ0 = ξ0[None,:]
    Δ = Δ[None,:]
    Iξξ = dA*np.einsum('ck,kw->cw',(ξ0)**2*combi,kernel) #(c,ω)
    Iξ = dA*np.einsum('ck,kw->cw',(ξ0)*combi,kernel) #(c,ω)
    Iii = dA*np.einsum('ck,kw->cw',(Δ.imag)**2*combi,kernel) #(c,ω)
    Iri = dA*np.einsum('ck,kw->cw',(Δ.imag*Δ.real)*combi,kernel) #(c,ω)
    Irr = dA*np.einsum('ck,kw->cw',(Δ.real)**2*combi,kernel) #(c,ω)

    #For the Q-integrals only one basis-function at a time is integrated:
    combi = combi[:3]*sign*np.sqrt(2) #only the first order basis-functions
    QRξ  = dA*np.einsum('ck,ka,kw->cwa',Δ.real*ξ0*combi,v3,kernel) #(c,ω,x/y)
    QIξ = dA*np.einsum('ck,ka,kw->cwa',Δ.imag*ξ0*combi,v3,kernel) #(c,ω,x/y)
    QR = dA*np.einsum('ck,ka,kw->cwa',Δ.real*combi,v3,kernel) #(c,ω,x/y)
    QI = dA*np.einsum('ck,ka,kw->cwa',Δ.imag*combi,v3,kernel) #(c,ω,x/y)
    
    #Ii = dA*np.einsum('ck,kw->cw',(Δ.imag)*combi,kernel) #(c,ω)
    #Ir = dA*np.einsum('ck,kw->cw',(Δ.real)*combi,kernel) #(c,ω)
    #Irξ = dA*np.einsum('ck,kw->cw',(Δ.real*ξ0)*combi,kernel) #(c,ω)
    #Iiξ = dA*np.einsum('ck,kw->cw',(Δ.imag*ξ0)*combi,kernel) #(c,ω)
    #
    #Qabs = dA*np.einsum('ka,kw->wa',abs(Δ[0,:,None])*v3,kernel) #(ω,x/y)
    Φintegral = dA*np.einsum('kw,ka,kb->wab',kernel*abs(Δ[0,:,None])**2,v3,v3)
    return np.array([Iξξ,Iξ,Iii,Iri,Irr]),np.array([QR,QI,QRξ,QIξ]),Φintegral#,Qabs,np.array([Ii,Ir,Irξ,Iiξ])