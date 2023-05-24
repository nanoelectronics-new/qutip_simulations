# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:32:46 2023

@author: jsaezmol
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

def H_static(B, g):
    """
    Generate Hamiltonian for a single spin qubit.

    Args:
        B: A 3-element list or numpy array containing the magnetic field vector in x, y, and z directions respectively.
        g: A 3-element list or numpy array containing the Lande g-factors in x, y, and z directions respectively.

    Returns:
        H: A numpy array containing the Hamiltonian.
    """
    # Constants
    mu_B = 1 #5.7883818060*10**-5 # eV/T

    # Hamiltonian
    omega = 0.5 * mu_B * np.array([g[0]*B[0], g[1]*B[1], g[2]*B[2]])
    
    Hz = omega[2] * sigmaz() 
    Hx = omega[0] * sigmax()
    Hy = omega[1] * sigmay()

    H = Hz + Hx + Hy
    
    return H


def H_driving_X(B_ac, g):
    """
    Generate Hamiltonian for a single spin qubit with driving along x-axis.

    Args:
        B_ac: A float value for ac magnetic field amplitude along x-axis.
        g: A 3-element list or numpy array containing the Lande g-factors in x, y, and z directions respectively.

    Returns:
        A numpy array containing the Hamiltonian for driving along x-axis.
    """
    
    # Constants
    mu_B =  1 #5.7883818060*10**-5 # eV/T
    h_bar = 1 #6.582*10**-16
    
    # Hamiltonian
    omega_ac = 0.5*mu_B*g*B_ac
    
    return omega_ac * sigmax()

def H_driving_Y(B_ac, g):
    """
    Generate Hamiltonian for a single spin qubit with driving along y-axis.

    Args:
        B_ac: A float value for ac magnetic field amplitude along y-axis.
        g: A 3-element list or numpy array containing the Lande g-factors in x, y, and z directions respectively.

    Returns:
        A numpy array containing the Hamiltonian for driving along x-axis.
    """
    
    # Constants
    mu_B =  1 #5.7883818060*10**-5 # eV/T
    h_bar = 1 #6.582*10**-16
    
    # Hamiltonian
    omega_ac = 0.5*mu_B*g*B_ac
    
    return omega_ac * sigmay()

def H_driving_cos_coefficient(t, args):
#     omega_MW = 2 # 
#     return args['a']*np.cos(args['omega_ac']*t)
    return np.cos(args['omega_MW']*t)

def H_driving_sin_coefficient(t, args):
#     omega_MW = 2 # 
#     return args['a']*np.cos(args['omega_ac']*t)
    return np.sin(args['omega_MW']*t)


def H_driving_cos_coefficient_Ramsey(t, args):
    if t>args['t_pihalf'] and t<args['t_wait']+args['t_pihalf']:
        return 0
    else:
        return np.cos(args['omega_MW']*t)

def H_driving_sin_coefficient_Ramsey(t, args):
    if t>args['t_pihalf'] and t<args['t_wait']+args['t_pihalf']:
        return 0
    else:   
        return np.sin(args['omega_MW']*t)



def calculate_expectation(H, g1, g2, t, args):
    """
    Calculate the expectation value of sigmaz over time using the given parameters.

    Parameters:
    -----------
    H : list
        The Hamiltonian of the system, represented as a list of operators.
    g1 : float
        The relaxation rate.
    g2 : float
        The dephasing rate.
    t : array_like
        The time steps to calculate the expectation value at.
    args : dict
        A dictionary of arguments to pass to the solver.

    Returns:
    --------
    result_list : array_like
        The expectation value of sigmaz over time.
    """
    # Initial state
    psi0 = basis(2, 1) # initialize state to ground state

    # collapse operators
    c_ops = []

    if g1 > 0.0:
        c_ops.append(np.sqrt(g1) * sigmap()) # Relax to ground state

    if g2 > 0.0:
        c_ops.append(np.sqrt(g2) * sigmaz()) # Dephase

    # Expectation values of the output
    e_ops = [sigmaz()]

    # Solve the evolution
    result = mesolve(H, psi0, t, c_ops, e_ops, args=args)  
    result_list = result.expect[0]

    return result_list