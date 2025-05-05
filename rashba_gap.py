import rashba as rs
import numpy as np
from numpy import sqrt, sin, cos, exp
from time import time as t
from IPython.display import clear_output
π=np.pi
τminus = np.array([
    [0,0],
    [1,0]
])
#A 2x2 Matrix of 4x4 Matrices:
#Entry (i,j) is a Matrix where only the (i,j) entries of the offdiagonal blocks are nonzero
τminus_matrix = np.zeros((2,2,4,4))
for i in range(2):
    for j in range(2):
        entry = np.zeros((2,2))
        entry[i,j]=1
        entry = np.kron(τminus,entry)
        τminus_matrix[i,j,...]=entry
def findGap(Δ_init=np.array([1,1,1,1,1,1,1]),N=50,V=np.array([-1,1,1,1,1,1,1]),maxiter=100,tol=1e-5,ωc=0.5,**kwargs):
    #Find energy-shell:
    kx_FS,ky_FS,plusShell,minusShell,mask = rs.findShells(N=N,ωc=ωc,**kwargs)
    sharedShell = plusShell & minusShell    #overlap of the two shells
    Nshell=len(kx_FS)
    dA = 1/N**2
    #A boolean filter-array indicating allowed pairings: 
    allowedPairing = np.zeros((len(V),2,2,Nshell),dtype=bool)
    allowedPairing[:,0,0,:]=plusShell[None,:]
    allowedPairing[:,1,1,:]=minusShell[None,:]
    allowedPairing[:,0,1,:]=sharedShell[None,:]
    allowedPairing[:,1,0,:]=sharedShell[None,:]
    #calculate the uncoupled interaction tensor as a function of k
    #Will be a (3,4,2,2,k) array, to account for all pairing combinationsd
    φ = rs.φvector(kx_FS,ky_FS,**kwargs) #(7,2,2,*k)-array
    #discount the entries of f corresponding to out-out-shell pairing
    φ[~allowedPairing]=0
    coupling = -V
    #Stretch the initial 2x2-gap-matrix to the size of the fermi-shell (uniformly)
    Δ=np.moveaxis(np.einsum('i,ijlK->jlK',Δ_init,φ),-1,0)
    #Δuniform = np.array([[1,1],[-1,1]])[None,...]+0*kx_FS[:,None,None]
    #BdG Hamiltonian on the interacting shell, using the initial gap-function
    H = rs.HBdG(kx_FS,ky_FS,Δ,**kwargs) #(Nshell,4,4)-array
    for i in range(maxiter):
        #Diagonalizing the BdG-Hamiltonian
        #print(np.round(Δ[0],3))
        E,V = np.linalg.eigh(H) #(Nshell,4) and (Nshell,4,4)-arrays
        Vdagger = V.conj().swapaxes(-1,-2)# (Nshell,4,4)
        #τminus_matrix gets transformed to the diagonal basis of the hamiltonian
        τminus_matrix_new = Vdagger[:,None,None,...]@ τminus_matrix[None,...] @ V[:,None,None,:,:] #(k,a,b,i,j)
        E = rs.fermi(E,**kwargs) #(Nshell,4)
        
        primesum = dA*np.einsum("mabK,Kj,Kabjj->m",φ,E,τminus_matrix_new)
        Δnew =np.einsum("mabK,m->Kab",φ,primesum*coupling)
        diff = abs(Δnew-Δ)
        # Check which gap-components flipped sign on the last iteration (real and )
        #flip = abs(Δnew+Δ)<tol
        #If a component of the gap completely flipped it's sign, it should be set to 0, so the algorithm converges
        #Otherwise, if some pairing is repulsive, the gap just flips sign on every iteration
        #For a bad choice of initial guesses, this could lead to some components becoming zero,
        #even if there is a finite solution.
        #Δnew[flip]=0
        print(f"Maximal difference is {np.max(diff)}")
        if i%10==0: clear_output(wait=True)
        if(np.all(diff<tol)):
            print(f"Converged after {i} iterations")
            #Algorithm converged, exit the iteration loop
            break
        else:
            #Update the order parameter and insert it into the BdG-Hamiltonian
            Δ=Δnew
            H[...,:2,2:] = Δ
            H[...,2:,:2] = Δ.conj().swapaxes(-2,-1)
        if i==maxiter-1:
            #If this get's printed, the result is likely incorrect
            print("Didn't converge, returning last value")
    returngap = np.zeros((N,N,2,2),dtype=complex)
    returngap[mask]=Δ
    #Return a 2D-grid of 2x2 matrices describing the gap,
    #and a 7 component vector of the basis-components
    return returngap,primesum*coupling
def free_energy(η,V,N=100,**kwargs):
    η_nonzero = η[V!=0]
    V_nonzero = V[V!=0]
    mf_term = np.sum(η_nonzero.conj()*η_nonzero/V_nonzero)
    kx,ky = rs.findShell(N=N,**kwargs)
    nk = len(kx)
    φ = rs.φsimplified(kx,ky)
    Δout = np.dot(φ,η)
    Δin = np.dot(φ.conj(),η)
    ξ = rs.ξsquarelattice(kx,ky,**kwargs)
    Eout = np.sqrt(ξ**2+abs(Δout)**2)#k-array
    Ein = np.sqrt(ξ**2+abs(Δin)**2)
    γ = np.linalg.norm(rs.γcoupling(kx,ky,**kwargs),axis=0)
    E = (γ+Eout,γ-Eout,Ein-γ,-Ein-γ)
    energyterm = 0
    for e in E:
        energyterm += np.sum(e[e<0])/N**2
    return mf_term.real+energyterm
def findGapFast(V,η0=np.ones(3),N=100,maxiter=400,tol=1e-8,quiet=False,q=np.zeros(2),**kwargs):
    """
    Gapfunction assuming ONLY interband pairing, in the pairing channels A1, E1 and E2.
    """
    kx,ky = rs.findShell(N=N,**kwargs)
    φ = rs.φsimplified(kx,ky)# (k,3)-array of basis-functions
    ξ = rs.ξsquarelattice(kx,ky,**kwargs)
    qv0 = 2*(q[0]*np.sin(kx)+q[1]*np.sin(ky))
    qvγ = np.dot(rs.v3(kx,ky,**kwargs),q)
    η = η0
    #Again, unsure:
    dA=1/len(kx)
    #flipper = np.array([-1,1,1])#For flipping the sign of s-wave component
    γ = np.linalg.norm(rs.γcoupling(kx,ky,**kwargs),axis=0)#k-array
    for i in range(maxiter):
        Δout=np.dot(φ,η)#k-array
        Δin = np.dot(φ.conj(),η)
        Eout = np.sqrt((ξ+qvγ)**2+abs(Δout)**2)#k-array
        Ein = np.sqrt((ξ-qvγ)**2+abs(Δin)**2)
        foutp = rs.fermi(Eout+γ+qv0,**kwargs)
        finp = rs.fermi(Ein-γ+qv0,**kwargs)
        foutm = rs.fermi(-Eout+γ+qv0,**kwargs)
        finm = rs.fermi(-Ein-γ+qv0,**kwargs)
        kernout = Δout*(foutp-foutm)/(2*Eout)#k-array
        kernin = Δin*(finp-finm)/(2*Ein)
        ηnew =-V* dA*((φ.transpose().conj()@kernout)+(φ.transpose()@kernin)) # (3,k)@k ->3
        diff = np.max(abs(ηnew-η))
        η=ηnew
        if not quiet:
            print(f"Maxdiff is {diff}")
            if ((i%10)==0): clear_output(wait=True)
        if(diff<tol):
            print(f"Converged after {i} iterations")
            #Algorithm converged, exit the iteration loop
            break
        elif i==maxiter-1:
            #If this get's printed, the result is likely incorrect
            print("Didn't converge, returning last value")
    return η

