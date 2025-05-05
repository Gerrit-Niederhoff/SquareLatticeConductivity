import numpy as np
from IPython.display import clear_output
from numpy import sqrt, sin, cos, exp, log
from numpy import pi as π
import square as sq
from time import time as t
import freeDispersion as fr
import math as m
def calculate_σ(V,η,ω,N=100,ωc=1,**kwargs):
    nonzero = (V!=0)
    coupling = -np.concatenate((V[nonzero],V[nonzero]))
    dof = len(coupling)    #Fluctuation degrees of freedom
    kx,ky = sq.findShell(ωc,N,**kwargs)
    #Restricting Delta from the full grid to the relevant shell
    #Δshell = Δ[mask]
    nk = len(kx)
    nω = len(ω)
    kchunks = m.ceil(nk/1e5)
    dA = 1/N**2
    #Splitting kx,ky and Delta into smaller chunks to save memory
    kx_intervals = np.array_split(kx,kchunks)
    ky_intervals = np.array_split(ky,kchunks)
    #Δ_intervals = np.array_split(Δshell,kchunks)
    σ_QP = np.zeros((nω,2,2))
    Qminus = np.zeros((nω,2,dof),dtype=complex)
    Q  = np.zeros((nω,2,dof),dtype=complex)
    Π  = np.zeros((nω,dof,dof),dtype=complex)
    for j in range(kchunks):
                print(f"Momentum interval {j+1} of {kchunks}")
                #clear_output(wait=True)
                σ_QP_chunk,Π_chunk,Q_chunk,Qm_chunk = calculate_chunk(
                    kx_intervals[j],
                    ky_intervals[j],
                    #Δ_intervals[j],
                    η,
                    ω,
                    dA,
                    nonzero,
                    **kwargs
                )
                σ_QP+=σ_QP_chunk
                Π+=Π_chunk
                Q+=Q_chunk
                Qminus+=Qm_chunk
    Ueffinv = np.diag(1/coupling)[None,...]-Π
    Ueff = np.linalg.inv(Ueffinv)
    σ_collective = np.einsum(
        'wap,wpq,wbq->wab',
        Q,Ueff,Qminus
    )
    σ_collective = (0.25j*σ_collective/ω[:,None,None]).real
    return σ_QP,σ_collective,Ueffinv,Q,Qminus
def calculate_chunk(kx,ky,η,ω,dA,nonzero_coupling,reg=1e-3j,**kwargs):
    """
    #Basic ingredients:
    s1 = t()
    velocity = sq.vel(kx,ky,**kwargs) #(*K,2)
    #velocity = np.array([[velocity,0*velocity],[0*velocity,velocity]]).transpose((2,3,0,1)) #(*k,x/y,2,2)
    Yarray = sq.Ytensor(kx,ky,choose=nonzero_coupling)   #(*Kdims,6,2,2)

    H = sq.HBdG(kx,ky,Δ,**kwargs)#(*Kdims,2,2)
    E,Vmatrix = np.linalg.eigh(H)     #(*Kdims,2) and (*Kdims,2,2), respectively
    Vdagger = Vmatrix.conj().swapaxes(-1,-2)

    #Basic quantities
    E_dif = E[...,:,None]-E[...,None,:] # (*Kdims,i,j)
    Fermi = sq.fermi(E,**kwargs)
    Fermi_dif = Fermi[...,:,None]-Fermi[...,None,:] #(*kdims,i,j)
    #Actually the product of propagator and the fermi-difference
    propagator = Fermi_dif[...,None,:,:]/(ω[None,:,None,None]+E_dif[...,None,:,:]+reg) #(*k,w,i,j)
    propagator_neg = Fermi_dif[...,None,:,:]/(-ω[None,:,None,None]+E_dif[...,None,:,:]-reg) #(*k,w,i,j)
    #Transform the basis-function-tensor into the diagonal basis
    Yarray = (Vdagger[:,None,...]) @ Yarray @ (Vmatrix[:,None,...]) #(*k,6,2,2)
    #Transform the velocity-array
    velocity = (Vdagger[:,None,...]) @ velocity @ (Vmatrix[:,None,...])#(*k,x/y,2,2)

    #Summing over momentum and other d.o.f's
    σ_quasiparticles = dA*np.einsum(
        "Kwij,Kaij,Kbji->wab",
        propagator,velocity,velocity
    )
    σ_quasiparticles = (1j*σ_quasiparticles/(ω[:,None,None])).real
    Π = dA/2*np.einsum(
        'Kwij,Kpqij->wpq',
        propagator,Yarray[:,:,None,:,:]*(Yarray[:,None,:,:,:].swapaxes(-1,-2))
    )    
    Q = dA*np.einsum(
        'Kwij,Kapij->wap',
        propagator,
        velocity[:,:,None,:,:]*(Yarray[:,None,:,:,:].swapaxes(-1,-2))
    )
    Qminus = dA*np.einsum(
        'Kwij,Kapij->wap',
        propagator_neg,
        velocity[:,:,None,:,:]*(Yarray[:,None,:,:,:].swapaxes(-1,-2))
    )
    e1 = t()
    """
    ###ALTERNATIVE APPROACH:
    s2=t()
    v3 = sq.v3(kx,ky,**kwargs)  #(*k,x/y)
    φ_array = sq.φvector(kx,ky,choose=nonzero_coupling) #(*k,dof/2)-array
    η_nonzero = η[nonzero_coupling]
    Δ = np.dot(φ_array,η_nonzero)
    ξbar,δ,Eplus,Eminus = sq.energy_terms(kx,ky,Δ,**kwargs) #All (*k)-arrays
    
    kernelnumerator = sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs)
    kernel = δ[:,None]*((ω[None,:]+reg)**2-4*(δ[:,None])**2)
    kernel_M = δ[:,None]*((-ω[None,:]-reg)**2-4*(δ[:,None])**2)
    kernel = kernelnumerator[:,None]/kernel #(*k,w)-array
    kernel_M = kernelnumerator[:,None]/kernel_M #(*k,w)-array
    Φ = -4*dA*np.einsum(
          'Kw,Ka,Kb->wab',
          kernel*abs(Δ[:,None])**2,v3,v3
    )
    Φ= (1j*Φ/(2*ω[:,None,None])).real

    
    
    #Defining the integrand WITHOUT THE FREQUENCY-FACTOR on the off-diagonal (to save memory)
    #Frequency needs to be appropriately multiplied in, after momentum is integrated out.
    ΠmatrixIntegrand1 = np.array([
          [-4*ξbar**2-4*(Δ.imag)**2,
           -4*Δ.real*Δ.imag],
          [-4*Δ.real*Δ.imag,
           -4*ξbar**2-4*(Δ.real)**2]
    ]).transpose((2,0,1))  #(*k,2,2)-array

    ΠmatrixIntegrand2 = np.array([
          [np.zeros_like(ξbar),
           2j*ξbar*(1)],
          [-2j*ξbar*(1),
           np.zeros_like(ξbar)]
    ]).transpose((2,0,1))  #(*k,2,2)-array

    dof = len(nonzero_coupling[nonzero_coupling])*2

    #ΠmatrixIntegrand = ΠmatrixIntegrand[:,:,:,None,:,None]*φ_arraysquared[:,None,None,:,None,:]    #(*k,w,2,dof/2,2,dof/2)
    #ΠmatrixIntegrand = ΠmatrixIntegrand.reshape(*np.shape(kernel),dof,dof) #(*k,w,dof,dof)-array

    Πalt = dA/2*np.einsum('Kipjq,Kw->wipjq',
        ΠmatrixIntegrand1[:,:,None,:,None]*φ_array[:,None,:,None,None]*φ_array[:,None,None,None,:],
        kernel)
    #Second array, containing terms that need to be multiplied with the frequency after integration:
    Πalt2 = dA/2*np.einsum('Kipjq,Kw->wipjq',
        ΠmatrixIntegrand2[:,:,None,:,None]*φ_array[:,None,:,None,None]*φ_array[:,None,None,None,:],
        kernel)
    #The missing frequency-factors:
    ωfactor = np.array([
          [0*ω+1,ω+reg],
          [ω+reg,0*ω+1]
    ]).transpose((2,0,1))#(w,2,2)
    #Adding both terms:
    Πalt = Πalt + Πalt2 * ωfactor[:,:,None,:,None]
    #Combining the mu and nu dimensions:
    Πalt = Πalt.reshape((len(ω),dof,dof))

    #Again, first defining everything without frequency-factors
    QvectorIntegrand1 = np.array(
          [4*Δ.real*ξbar,
           4*Δ.imag*ξbar]).transpose((1,0))#(*k,2)
    QvectorIntegrand1 = QvectorIntegrand1[:,None,:,None]*v3[:,:,None,None]*φ_array[:,None,None,:] #(*k,x/y,2,dof/2)
    
    QvectorIntegrand2 = np.array(
          [-2j*Δ.imag*(1),
           -2j*Δ.real*(1)]).transpose((1,0))#(*k,2)
    QvectorIntegrand2 = QvectorIntegrand2[:,None,:,None]*v3[:,:,None,None]*φ_array[:,None,None,:] #(*k,x/y,2,dof/2)
    
    #Terms that aren't proportional to frequency
    Qalt = dA*np.einsum('Kaip,Kw->waip',QvectorIntegrand1,kernel)
    Qalt_M = dA*np.einsum('Kaip,Kw->waip',QvectorIntegrand1,kernel_M)
    #Terms that are proportional to frequency:
    Qalt_f = dA*np.einsum('Kaip,Kw->waip',QvectorIntegrand2,kernel)
    Qalt_M_f = dA*np.einsum('Kaip,Kw->waip',QvectorIntegrand2,kernel_M)

    #Re-introducing the missing frequency-factors:  
    ωfactor = np.array([ω+reg,ω+reg]).transpose((1,0)) #(w,2)
    Qalt = Qalt+Qalt_f*ωfactor[:,None,:,None]
    #For negative frequencies:
    ωfactor = np.array([-ω-reg,-ω-reg]).transpose((1,0)) #(w,2)
    Qalt_M = Qalt_M+Qalt_M_f*ωfactor[:,None,:,None]

    #Reshaping
    Qalt = Qalt.reshape((len(ω),2,dof))
    Qalt_M = Qalt_M.reshape((len(ω),2,dof))#(*k,x/y,dof)
    e2 = t()
    """
    #COMPARING:

    print("Quasiparticle conductivity difference: ",np.max(abs(Φ-σ_quasiparticles)))
    print("Π00 difference: ",np.max(abs(Π-Πalt)[:,0:5,0:5]))
    print("Π10 difference: ",np.max(abs(Π-Πalt)[:,5:,0:5]))
    print("Π01 difference: ",np.max(abs(Π-Πalt)[:,0:5,5:]))
    print("Π11 difference: ",np.max(abs(Π-Πalt)[:,5:,5:]))
    print("Q difference: ",np.max(abs(Q-Qalt)))
    print("Q_m difference: ",np.max(abs(Qminus-Qalt_M)))
    print("Original method: ",e1-s1," seconds.")
    print("New method: ", e2-s2, " seconds.")
    """
    return Φ,Πalt,Qalt,Qalt_M
    #return σ_quasiparticles,Π,Q,Qminus
def collective_σ(Ueffinv,Q,Qminus,ω):
    Ueff = np.linalg.inv(Ueffinv)
    σ_collective = np.einsum(
        'wap,wpq,wbq->wab',
        Q,Ueff,Qminus
    )
    σ_collective=(0.25j*σ_collective/ω[:,None,None]).real
    return σ_collective

####Alternative functions, specifically for ONLY s+id order parameters
#Calculate 9 Integrals and build everything from those
def σ_simplified(Vs,Vd,ηs,ηd,ω,**kwargs):
    """
    Calculate the optical conductivity of the simple one-band model. Takes scalars Vs,Vd,ηs,ηd
    and an array of frequencies ω. All additional keyword arguments are passed on to polarizationbubbles.
    """
    Vd*=-1
    Vs*=-1
    Φ,Π,Q,Qm =polarizationBubbles(ηs,ηd,ω,**kwargs)
    σQP = -0.5*(Φ/ω[:,None,None]).imag
    VeffInv = np.diag([1/Vd,1/Vs,1/Vd,1/Vs])[None,:,:]-Π/2
    Veff = np.linalg.inv(VeffInv)
    σ_collective = np.einsum('wap,wpq,wbq->wab',Q,Veff,Qm)
    σ_collective = -0.25*(σ_collective/ω[:,None,None]).imag
    return σQP,σ_collective,VeffInv,Q,Qm
def σ_bilayer(Vs,Vd,ηs,ηd,ω,J=0,μ=0,**kwargs):
    """
    Calculate conductivity for the bilayer system. Vs and Vd should be the 2x2-pairing matrices, 
    ηs,ηd the corresponding 2-vector gaps. ω should be an array of frequencies as usual.
    Returns 3 tuples, for band 1, band 2 and the interband-contribution, respectively.
    Each Tuple contains the optical conductivity, the inverse effective coupling, Q(ω) and Q(-ω)
    for the respective contribution.
    """
    #Construct the V matrices:
    VarrInv = np.linalg.inv(np.array([-Vd,-Vs])) #(2,2,2)-array
    pairing11 = np.kron(np.eye(2),np.diag(VarrInv[:,0,0]))
    pairing22 = np.kron(np.eye(2),np.diag(VarrInv[:,1,1]))
    pairing12 = np.kron(np.eye(2),np.diag(VarrInv[:,0,1]))
    pairing21 = np.kron(np.eye(2),np.diag(VarrInv[:,1,0]))
    #Varr = np.concatenate((Varr,Varr),axis=0) #μ,α,β
    pairingBlock = np.array([
         [pairing11,pairing12],
         [pairing21,pairing22]
    ]).transpose((0,2,1,3)).reshape(8,8)
    #Calculate the polarizationbubbles via integration over the BZ. Each Band is simply a cosine-band
    #shifted by +-J, so the previous functions can be reused.
    if type(ηs[0])!=np.complex128:
        #If the gap components are real numbers, it is assumed that the gap is of s+id type
        Φ1, Π1, Q1, Qm1 = polarizationBubbles(ηs[0],ηd[0],ω,μ=μ+J,**kwargs)
        Φ2, Π2, Q2, Qm2 = polarizationBubbles(ηs[1],ηd[1],ω,μ=μ-J,**kwargs)
    else:
        #For a general gap, the components are complex. Then, the general function is used.
        Φ1, Π1, Q1, Qm1 = complexPolarizationBubbles(ηs[0],ηd[0],ω,μ=μ+J,**kwargs)
        Φ2, Π2, Q2, Qm2 = complexPolarizationBubbles(ηs[1],ηd[1],ω,μ=μ-J,**kwargs)
    o=np.zeros_like(Π1)
    Π =  np.array([
         [Π1,o],
         [o,Π2]
    ]).transpose((2,0,3,1,4)).reshape((len(ω),8,8))
    Q = np.concatenate((Q1,Q2),axis=-1)
    Qm = np.concatenate((Qm1,Qm2),axis=-1)
    #Calculate the intraband contributions:
        #Band 1:
    σQP1 = -0.5*(Φ1/ω[:,None,None]).imag
        #Band 2:
    σQP2 = -0.5*(Φ2/ω[:,None,None]).imag
    #Calculate the fluctuation/interband contributions:
    #Creating a block matrix: 8x8, with the outer block corresponding to bands a and b
    #Π = np.einsum('mga,nbg,gwmn->wambn',Varr,Varr,Π).reshape((len(ω),8,8))
    
    
    VeffInv = pairingBlock[None,...]-Π/2
    #transpose moves rows of outer matrix in front of rows for inner matrix, so reshape has the desired effect
    Veff = np.linalg.inv(VeffInv)

    #Creating the updated Q-vector:
    #Q = np.einsum('mga,gwjm->wjam',Varr,Q).reshape((len(ω),2,8))
    #Qm = np.einsum('mag,gwjm->wjam',Varr,Qm).reshape((len(ω),2,8))
    σ_coll = np.einsum('wap,wpq,wbq->wab',Q,Veff,Qm)
    σ_coll = -0.25*(σ_coll/ω[:,None,None]).imag

    return σQP1+σQP2,σ_coll,VeffInv,Q,Qm

#Calculate basic Integrals and build everything from those
def polarizationBubbles(ηs,ηd,ω,N=100,ωc=20,reg=1e-4j,use="square",**kwargs):
    """
    Calculates the 4 polarization bubbles of one cosine-band.
    ηs,ηd are real scalars describing the amplitude of s and id gap-components.
    ω is an array of frequencies.
    Returns Φ(ω): (ω,x/y,x/y)-array (last 2 axes are conductivity-components)
    Π(ω): (ω,4,4)-array, last 2 axes are fluctuation-components (d,s,id,is)
    Q(ω) and Q(-ω): (ω,x/y,4)-array. Second axis is spatial direction, 3rd axis are fluctuations.
    This function uses formulas simplified specifically for an s+id gap, which is why the gap components
    are treated as real. For a more general function, see complexPolarizationBubbles, 
    which yields equivalent results but takes a little longer, due to treating the components as complex numbers.
    """
    if use=="square":
        file = sq
    elif use=="free":
         file = fr
    else: raise ValueError("Only able to use 'free' or 'square'.")
    ωreg = ω+reg
    kx,ky = file.findShell(ωc,N,**kwargs)
    nk = len(kx)
    nω = len(ω)
    kchunks = m.ceil(nk/2e5)
    dA = 1/N**2
    #Splitting kx,ky and Delta into smaller chunks to save memory
    kx_intervals = np.array_split(kx,kchunks)
    ky_intervals = np.array_split(ky,kchunks)

    integrals = np.zeros((8,nω),dtype=complex) #Stores the 5 scalar integrals
    vector_integrals = np.zeros((6,nω,2),dtype=complex) #Stores the 4 vector integrals
    Φintegral = np.zeros((nω,2,2),dtype=complex)
    Ld = 0
    Ls = 0
    Ld1 =0
    for j in range(kchunks):
        print(f"Momentum interval {j+1} of {kchunks}")
        #clear_output(wait=True)
        Ichunk,vIchunk,Φchunk,LdChunk,LsChunk,Ld1Chunk=chunk_simplified(
              kx_intervals[j],ky_intervals[j],ηs,ηd,ωreg,dA,file,**kwargs
         )
        integrals+=Ichunk
        vector_integrals+=vIchunk
        Φintegral+=Φchunk
        Ld+=LdChunk
        Ls+=LsChunk
        Ld1+=Ld1Chunk
    #Construct relevant objects:
    Φintegral*=-4
    I0, I2, I4, Iξd, Iξ, Iξdd,I1,I3 = integrals
    Q1, Q2, Q3, Q4, Q5, Q6 = vector_integrals #all (w,2)
    o = np.zeros_like(I4)
    #Inserting results from analytical simplifications.
    #The function complexPolarizationBubbles is more readable, and more general
    #  (but a little slower) in these calculations
    Π = -2*np.array([
         [-Ld+(ωreg**2/2-2*ηs**2)*I2,
          -Ld1+(ωreg**2/2-2*ηs**2)*I1,
          -1j*ωreg*Iξdd+2*ηs*ηd*I3,
          2*ηs*ηd*I2-1j*ωreg*Iξd],

         [-Ld1+(ωreg**2/2-2*ηs**2)*I1,
          -Ls+(ωreg**2/2-2*ηs**2)*I0,
          2*ηs*ηd*I2-1j*ωreg*Iξd,
          -1j*ωreg*Iξ+2*ηs*ηd*I1],

         [1j*ωreg*Iξdd+2*ηs*ηd*I3,
          2*ηs*ηd*I2+1j*ωreg*Iξd,
          -Ld+ωreg**2/2*I2-2*ηd**2*I4,
          -Ld1+ωreg**2/2*I1-2*ηd**2*I3],

         [2*ηs*ηd*I2+1j*ωreg*Iξd,
          1j*ωreg*Iξ+2*ηs*ηd*I1,
          -Ld1+ωreg**2/2*I1-2*ηd**2*I3,
          -Ls+ωreg**2/2*I0-2*ηd**2*I2]
    ]).transpose(2,0,1)
    Q = np.array([
         4*ηs*Q5-2j*ωreg[:,None]*ηd*Q1,
         4*ηs*Q2-2j*ωreg[:,None]*ηd*Q6,
         -4*ηd*Q3-2j*ωreg[:,None]*ηs*Q6,
         -4*ηd*Q5-2j*ωreg[:,None]*ηs*Q4,
    ]).transpose(1,2,0)
    Qm = np.array([
         4*ηs*Q5+2j*ωreg[:,None]*ηd*Q1,
         4*ηs*Q2+2j*ωreg[:,None]*ηd*Q6,
         -4*ηd*Q3+2j*ωreg[:,None]*ηs*Q6,
         -4*ηd*Q5+2j*ωreg[:,None]*ηs*Q4,
    ]).transpose(1,2,0)
    return Φintegral,Π,Q,Qm
def chunk_simplified(kx,ky,ηs,ηd,ω,dA,file,**kwargs):
    """
    Calculates all the relevant integrals. The frequency-array should already contain +iδ.
    """
    v3 = file.v3(kx,ky,**kwargs)
    φd = file.φvector(kx,ky,choose=np.array([False,True,False,False,False]))[:,0] #(*k,dof/2)-array
    Δ = φd*1j*ηd+ηs
    ξbar,δ,Eplus,Eminus = file.energy_terms(kx,ky,Δ,**kwargs) #All (*k)-arrays
    basickernel = (file.fermi(Eplus,**kwargs)-file.fermi(Eminus,**kwargs))/δ
    kernel = basickernel[:,None]/((ω[None,:])**2-4*(δ[:,None])**2)  #(k,w)
    I0 = dA*np.einsum('kw->w',kernel)   #w
    I2 = dA*np.einsum('kw,k->w',kernel,φd**2)   #w
    I4 = dA*np.einsum('kw,k->w',kernel,φd**4)   #w
    Iξd = dA*np.einsum('kw,k->w',kernel,ξbar*φd)
    Iξ = dA*np.einsum('kw,k->w',kernel,ξbar)
    Iξdd = dA*np.einsum('kw,k->w',kernel,ξbar*φd**2)
    Ld1 = dA*np.sum(φd*basickernel)/2
    I1 = dA*np.einsum('kw,k->w',kernel,φd)
    I3 = dA*np.einsum('kw,k->w',kernel,φd**3)
    Ld = dA*np.sum(φd**2*basickernel)/2
    Ls = dA*np.sum(basickernel)/2
    Φintegral = dA*np.einsum('kw,ka,kb->wab',kernel*abs(Δ[:,None])**2,v3,v3)
    Qint1 = dA*np.einsum('kw,ka->wa',kernel*(φd[:,None])**2,v3)
    Qint2 = dA*np.einsum('kw,ka->wa',kernel*(ξbar[:,None]),v3)
    Qint3 = dA*np.einsum('kw,ka->wa',kernel*(φd**2*ξbar)[:,None],v3)
    Qint4 = dA*np.einsum('kw,ka->wa',kernel,v3)
    Qint5 = dA*np.einsum('kw,ka->wa',kernel*(φd*ξbar)[:,None],v3)
    Qint6 = dA*np.einsum('kw,ka->wa',kernel*(φd)[:,None],v3)
    return np.array([I0,I2,I4,Iξd,Iξ,Iξdd,I1,I3]),np.array([Qint1,Qint2,Qint3,Qint4,Qint5,Qint6]),Φintegral,Ld,Ls,Ld1
def complexPolarizationBubbles(ηs,ηd,ω,N=100,ωc=20,reg=1e-4j,use="square",**kwargs):
    """
    Calculates the 4 polarization bubbles of one cosine-band, for the case where 
    ηs,ηd may be complex scalars.
    ω is an array of frequencies.
    Returns Φ(ω): (ω,x/y,x/y)-array (last 2 axes are conductivity-components)
    Π(ω): (ω,4,4)-array, last 2 axes are fluctuation-components (d,s,id,is)
    Q(ω) and Q(-ω): (ω,x/y,4)-array. Second axis is spatial direction, 3rd axis are fluctuations.
    """
    print("Interpreting ηs,ηd as complex amplitudes!")
    ωreg = ω+reg
    kx,ky = sq.findShell(ωc,N,**kwargs)
    nk = len(kx)
    nω = len(ω)
    kchunks = m.ceil(nk/5e4)
    dA = 1/N**2
    #Splitting kx,ky and Delta into smaller chunks to save memory
    kx_intervals = np.array_split(kx,kchunks)
    ky_intervals = np.array_split(ky,kchunks)
    integrals = np.zeros((5,3,nω),dtype=complex)
    vectorIntegrals = np.zeros((4,2,nω,2),dtype=complex)
    Φintegral = np.zeros((nω,2,2),dtype=complex)
    for j in range(kchunks):
        print(f"Momentum interval {j+1} of {kchunks}")
        #clear_output(wait=True)
        Ichunk,vIchunk,Φchunk=complexChunks(
              kx_intervals[j],ky_intervals[j],ηs,ηd,ωreg,dA,**kwargs
         )
        integrals+=Ichunk
        vectorIntegrals+=vIchunk
        Φintegral+=Φchunk
    Φintegral*=-4
    Iξξ,Iξ,Iii,Iri,Irr = integrals
    QR,QI,QRξ,QIξ = vectorIntegrals
    
    #Creating the ν-indices of Π first (to later become the outer indices of the 4x4 matrix)
    Π = np.array([
         [-4*Iξξ-4*Iii,2j*ωreg*Iξ-4*Iri],
         [-2j*ωreg*Iξ-4*Iri,-4*Iξξ-4*Irr]
    ]).transpose((2,0,1,3))#(3,ν,ν',w)
    
    #The first axis contains the combinations d^2,ds,s^2
    #Arrange these combinations appropriately into the μ,μ'-indices.
    #Then rearrange: Frequency to the front, then ν,μ,ν',μ', so that reshape fuses the appropriate axes
    Π = np.array([
         [Π[0],Π[1]],
         [Π[1],Π[2]]
    ]).transpose((4,2,0,3,1)).reshape(nω,4,4)
    #Similarly for Q. Here however, the component-axis contains only d and s. This means that one can reshape right-away
    Q = np.array([
         4*QRξ-2j*ωreg[None,:,None]*QI,
         -4*QIξ-2j*ωreg[None,:,None]*QR
    ]).transpose((2,3,0,1)).reshape(nω,2,4)
    Qm = np.array([
         4*QRξ+2j*ωreg[None,:,None]*QI,
         -4*QIξ+2j*ωreg[None,:,None]*QR
    ]).transpose((2,3,0,1)).reshape(nω,2,4)
    return Φintegral,Π,Q,Qm
def complexChunks(kx,ky,ηs,ηd,ω,dA,**kwargs):
    """
    Calculates all the relevant integrals for complex s,d components. The frequency-array should already contain +iδ.
    """
    v3 = sq.v3(kx,ky,**kwargs)
    φd = sq.φvector(kx,ky,choose=np.array([False,True,False,False,False]))[:,0] #(*k)-array
    Δ = ηd*φd+ηs
    ξbar,δ,Eplus,Eminus = sq.energy_terms(kx,ky,Δ,**kwargs) #All (*k)-arrays
    basickernel = (sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs))/δ
    kernel = basickernel[:,None]/((ω[None,:])**2-4*(δ[:,None])**2)  #(k,w)
    φcombinations = np.array([φd**2,φd,np.ones_like(φd)])
    ξbar = ξbar[None,:]
    Δ = Δ[None,:]
    Iξξ = dA*np.einsum('ck,kw->cw',(ξbar)**2*φcombinations,kernel) #(c,ω)
    Iξ = dA*np.einsum('ck,kw->cw',(ξbar)*φcombinations,kernel) #(c,ω)
    Iii = dA*np.einsum('ck,kw->cw',(Δ.imag)**2*φcombinations,kernel) #(c,ω)
    Iri = dA*np.einsum('ck,kw->cw',(Δ.imag*Δ.real)*φcombinations,kernel) #(c,ω)
    Irr = dA*np.einsum('ck,kw->cw',(Δ.real)**2*φcombinations,kernel) #(c,ω)
    φcombinations = φcombinations[1:]
    QRξ  = dA*np.einsum('ck,ka,kw->cwa',Δ.real*ξbar*φcombinations,v3,kernel) #(c,ω,x/y)
    QIξ = dA*np.einsum('ck,ka,kw->cwa',Δ.imag*ξbar*φcombinations,v3,kernel) #(c,ω,x/y)
    QR = dA*np.einsum('ck,ka,kw->cwa',Δ.real*φcombinations,v3,kernel) #(c,ω,x/y)
    QI = dA*np.einsum('ck,ka,kw->cwa',Δ.imag*φcombinations,v3,kernel) #(c,ω,x/y)
    Φintegral = dA*np.einsum('kw,ka,kb->wab',kernel*abs(Δ[0,:,None])**2,v3,v3)
    return np.array([Iξξ,Iξ,Iii,Iri,Irr]),np.array([QR,QI,QRξ,QIξ]),Φintegral