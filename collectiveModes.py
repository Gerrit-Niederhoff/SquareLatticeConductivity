import numpy as np
from IPython.display import clear_output
from numpy import sqrt, sin, cos, exp, log
from numpy import pi as π
from scipy.optimize import minimize_scalar
import square as sq
from time import time as t
import math as m
def swave_BSfactor(ω,Vd,ηs,N=100,reg=1e-4j,**kwargs):
    ωreg=ω+ reg
    kx,ky,mask = sq.findShell(20,N,**kwargs)
    dA = 1/N**2
    #Splitting kx,ky and Delta into smaller chunks to save memory
    φd = sq.φvector(kx,ky,
                    choose=np.array([False,True,False,False,False]))[:,0]
    ξbar,δ,Eplus,Eminus = sq.energy_terms(kx,ky,ηs,**kwargs)
    basickernel = (sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs))/δ
    kernel = basickernel/(ωreg**2-4*δ**2)  #(k,w)
    Ld = dA*np.sum(φd**2*basickernel/2)
    I2 = dA*np.sum(φd**2*kernel)
    return -1/Vd-Ld+ωreg**2/2*I2
def findMode(ηs,ηd,Vd,swave=True,**kwargs):
    if swave:
        dummy = lambda ω: abs(swave_BSfactor(ω,Vd,ηs,**kwargs))
        ω0 = 0
    else:
        ω0 =2*ηs
        dummy = lambda ω: abs(mixedwave_factor(ω,ηs,ηd,**kwargs))
    return minimize_scalar(dummy,ω0,bounds=(0,2*ηs))
def mixedwave_factor(ω,ηs,ηd,N=100,reg=1e-4j,**kwargs):
    ωreg=ω+ reg
    kx,ky,mask = sq.findShell(20,N,**kwargs)
    dA = 1/N**2
    #Splitting kx,ky and Delta into smaller chunks to save memory
    φd = sq.φvector(kx,ky,
                    choose=np.array([False,True,False,False,False]))[:,0]
    ξbar,δ,Eplus,Eminus = sq.energy_terms(kx,ky,ηs+1j*ηd*φd,**kwargs)
    basickernel = (sq.fermi(Eplus,**kwargs)-sq.fermi(Eminus,**kwargs))/δ
    kernel = basickernel/(ωreg**2-4*δ**2)  #(k,w)
    I0 = dA*np.sum(kernel)
    I2 = dA*np.sum(kernel*φd**2)
    I4 = dA*np.sum(kernel*φd**4)
    return I0*(ωreg**2/2-2*ηs**2)*(ωreg**2/2*I2-2*ηd**2*I4)-(2*ηs*ηd*I2)**2