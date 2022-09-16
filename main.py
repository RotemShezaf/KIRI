# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np  # math functions
import scipy  # scientific functions
import matplotlib.pyplot as plt  # for plotting figures and setting their properties
import pandas as pd  # handling data structures (loaded from files)
from scipy.stats import linregress  # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit  # non-linear curve fitting
from matplotlib.offsetbox import AnchoredText
from scipy.constants import mu_0
from derivative import dxdt
import math


d1 = 80e-3  # m
d2 = 38e-3  # m
𝑛1 = 518
𝑛2 = 1500
d1_wire = 5e-4  # m
d2_wire = 2e-4  # m
p_copper = 1.72e-8

h1= 0.300
h2= 0.048
h2_err = 0.002

L_sec = 84.3e-3  # H
L_sec_err = 0.1e-3  # H

R_sec = 124.9  # ohm
R_sec_err = 0.1  # ohm

I = 0.595 #A
dI = 0.001 #A

d3_wire = 3e-3 
n3 = 76 #𝑚 Number of turns around the oven
d3 = 0.0167  #𝑚 Diameter of the oven
p3 = 1.45e-6  #Ω⋅𝑚 Wire resistivity, 𝜌𝑟
p_m = 7.1e3 #𝑘𝑔/𝑚3  AB wire mass density, 𝜌𝑚
c = 510  #𝐽/(𝑘𝑔⋅𝐾 ) ⁄ Specific heat capacity, 𝑐𝑝  

def R_L(N, d_wire, d, p):
     return p*N*(np.pi*d)/(np.pi*(d_wire/2)**2)

def L_l(N, d, h):
    return mu_0*N**2*np.pi*(d/2)**2/h
    
def B_L(N, d, h, I):
    return (mu_0*N*I)/np.sqrt(h**2 + 4*(d/2)**2)

def linearCurve(x,a,b) :
    return a*x + b

def cool_reg(t,a,b):
    return a*np.exp(-b*t)

R2 =  R_L(n2, d2_wire, d2,  p_copper)
print( "R secondary:", R2)
L2 = L_l(n2, d2, h2)
print( "L secondary:", L2)

R3 = R_L(n3, d3_wire, d3,  p_copper)
m3 = n3*np.pi*d3*np.pi*(d3_wire/2)**2*p_m 

P = I**2*R3


V_coil1 = 1.6324 # volt
V_coil2 = 1.8705 #volt
err_V = 0.0002
shanai = V_coil1/V_coil2
shanai_err = (V_coil1/V_coil2)*np.sqrt((err_V/V_coil1)**2 + (err_V/V_coil2)**2)
print(shanai, "+-", shanai_err)
phase = 10.8 #degrees
err_phase = 0.1
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

data = pd.read_csv("VvsF_Vin_200mV.csv", header=0)
measurements = [(data[f"Signal Frequency {i}"], data[f"RMS Voltage {i}"]) for i in range(1, 4)]




plt.figure()
plt.grid()
plt.semilogx()

plt.xlabel(r"$f\ [Hz]$")
plt.ylabel(r"$V\ [V]$")

for i, (f, V) in enumerate(measurements):
    plt.plot(f, V, ".", label=f"$V_{{s{i}}}$")

def rotemDiff( func, n):
    result = np.array([])
    for i in range(n , func.size -n -1):
        result = np.append(result, func[i+n] - func[i - n])
    return result

def derivative(T, V , T1, T2, t, t2):


    dV = V.diff( )
    dT = T.diff( )
    dV_dT  = dxdt(V, T, kind="finite_difference", k=20)
    plt.figure()
    plt.plot( T, dT, '.', label = "dT" )
    plt.legend()
    plt.grid()
    inds = (T > T1) & (T < T2)
    plt.figure()
    plt.plot( T[inds],  dV_dT[inds], '.', label = "dT" )
    print( "kiri temperature :", T[np.argmin(dV_dT)])#247.4 בשני
    plt.legend()
    plt.grid()

def analayze(csv, t1, t2, T1, T2):
    V_err = 0.01
    T_err = 0.01
    data = pd.read_csv( csv, header=0)
    t = data["Time (sec) "]
    T = data["  Temperature (C) "]
    V = data["  RMS Voltage Ch2 (v) "]


    plt.figure()
    plt.grid()

    plt.xlabel(r"$t\ [s]$")
    plt.ylabel(r"$T\ [C]$")
    plt.errorbar(t, T, xerr=1, yerr= V_err, fmt='+', zorder=1, label="error bar")
    plt.plot(t, T, ".", label="Temperature")
    inds = ( t < t1)
    reg = linregress(  t[inds], T[inds] )
    plt.plot( t[inds], linearCurve(  t[inds], reg.slope, reg.intercept), label="regression heat part")
    inds2 = ( t > t2 )
    fit = cfit(  cool_reg, t[inds2] - t2 ,   T[inds2] )
    plt.plot( t[inds2] , cool_reg(  t[inds2] - t2, fit[0][0], fit[0][1]), label="regression cool part")
    plt.legend()
    plt.figure()
    plt.plot( T, V, '.', label = "V")
    plt.legend()
    plt.grid()
    derivative(T, V, T1, T2, t, t2)


analayze("N1_1500Hz_35V.csv", t1 = 713, t2 = 725, T1 = 138, T2 = 154)
analayze("N2_1500Hz_35V.csv", t1 = 669, t2 = 690, T1 = 212, T2 = 264)
plt.show()
