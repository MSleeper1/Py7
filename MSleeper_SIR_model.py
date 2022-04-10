#! /usr/bin/python
#
##################################################################################################
# SIR.py
# 
# The SIR epidemic model, modified from https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
#
# A simple mathematical description of the spread of a disease in a population is the so-called SIR model, which divides the 
# (fixed) population of N individuals into three "compartments" which may vary as a function of time, t:
#
#   S(t)    those SUSCEPTIBLE but not yet infected with the disease;
#   I(t)    the number of INFECTIOUS individuals;
#   R(t)    those individuals who are REMOVED, either by recovering from the disease and now being immune, or by dying.
#
# The SIR model describes the change in the (constant) population N = S+I+R of each of these compartments in terms of two parameters, β
# and γ. β describes the effective contact rate of the disease: an infected individual comes into contact with βN other
# individuals per unit time (of which the fraction that are susceptible to contracting the disease is S/N)  The typical time
# between contacts is 1/β.
# γ is the mean recovery rate: that is, 1/γ is the mean period of time during which an infected individual can pass it on.
#
# The differential equations describing this model were first derived by Kermack and McKendrick [Proc. R. Soc. A, 115, 772 (1927)]:
#
#   dS/dt = −βIS/N         
#   dR/dt =  γI
#   dI/dt =  βIS/N − γI    (# of new infections - # removed)
#
# The basic reproduction number (ratio) Ro is:
#
#    Ro = β/γ      the expected number of new infectious from a single infection in a susceptible population
#
# Note that:
#
#   dI/dt = I(βS/N − γ)
#
# thus, when S = Nγ/β, dI/dt = 0 and the infection will peak.  Furthermore, if βS/N > γ, the rate of infection is positive (increases):  N < β/γS   or S > N/Ro.
# Thus, if S < N/Ro, the infection may never spread.  This is the basis of herd vaccination.  
#
# created - G. Wolfe
# Modified - M. Sleeper 4/7
##################################################################################################


#############################
#### Module dependencies ####
#############################

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#############################
#### Global variables ####
#############################

NUM_DAYS = 160

N = 1000                      # Total population, N.
I0, R0 = 1, 0                 # Initial number of infected and recovered individuals, I0 and R0.


# beta is contact rate per fraction of population that an infected individual comes into contact with 1/time (days); 1/beta = days before transmit
beta = 1/5
beta_list = []

# gamma is mean recovery rate in 1/time (days); 1/gamma = infection duration
gamma = 1/20  
gamma_list = []                

#############################
#### Variables and Names ####
#############################


###################
#### Functions ####
###################

#---------------------------------------------
# The SIR model differential equations.
# here y = S, I, or R, and we calculate dS/dt, dI/dt, and dR/dt
# use global N, beta, and gamma (constants here)
# if these are allowed to vary, must pass as arguments
# for example, if N can vary: (odeint(deriv,y0,t, args=(N))
#---------------------------------------------

def deriv(y, t):
    S, I, R = y

    dSdt = -beta*S*I/N
    dRdt = gamma*I
    dIdt = beta*S*I/N - gamma*I
    #dIdt = (beta*S/N) - gamma*I        #Additional way to write calculation for dIdt
    #dIdt = -dSdt - dRdt                #Additional way to write calculation for dIdt
    
    return dSdt, dIdt, dRdt


##############
#### Main ####
##############

#Create list of all gamma and beta values to be tested
for i in range(1, 21):
    beta_list.append(1/i)
    gamma_list.append(1/i)

#Creating CSV files to store data for both beta variation and gamma variation
beta_variation_file = open('SIR_beta_variation.csv', 'w')
beta_variation_file.write("contact time in days (1/beta), incubation time in days (1/gamma), Ro (beta/gamma) \n") 
beta_variation_file.close()

gamma_variation_file = open('SIR_gamma_variation.csv', 'w')
gamma_variation_file.write("contact time in days (1/beta), incubation time in days (1/gamma), Ro (beta/gamma) \n") 
gamma_variation_file.close()

######-----------------------------#######
######  Beta values varying MAIN   #######
######-----------------------------#######

#Iterate over beta list values while keeping gamma value constant. 
for value in beta_list:
    beta = value

    # Step 1: set the initial conditions and add data parameters to CSV file

    beta_variation_file = open('SIR_beta_variation.csv', 'a')
    beta_variation_file.write("{}, {}, {:.1f}\n".format( 1/beta, 1/gamma, beta/gamma ) )
    beta_variation_file.close()

    S0 = N - I0 - R0 
    t = np.arange(NUM_DAYS)
    y0 = (S0, I0, R0)

    # Step 2: integrate the SIR equations over the time grid, t.

    y_array = odeint(deriv, y0, t) #ode returns columns and we need to convert to rows
    S, I, R = y_array.T #T will transpose our array (matrix)¡¡

    # Step 3: plot the data on three separate curves for S(t), I(t) and R(t)

    fig, ax = plt.subplots()
    ax.set_facecolor('gainsboro')

    ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/N, 'r', alpha=0.5, lw=3, label='Infectious')
    ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Removed')

    ax.set_xlabel('time(days)')
    ax.set_ylabel('fraction of population')

    #add legend
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    #add gridlines
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='w', lw=1, ls='-')

    #turn off bounding box
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    # make up the text box to display the parameters
    textstr = '\n'.join((
        'contact time = {} days'.format(1/beta), 'incubation time = {} days'.format(1/gamma),'R0 = {:.1f}'.format(beta/gamma) ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper right in axes coords

    ax.text(0.95, 0.8, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)

    # Save plot in directory folder with name of value varied and value
    plt.savefig('beta_{:.0f}_days.png'.format(1/beta))
    plt.close()



######-----------------------------#######
######  Gamma values varying MAIN  #######
######-----------------------------#######

#Reset beta value
beta = 1/5

#Iterate over gamma list values while keeping gamma value constant
for value in gamma_list:
    gamma = value

    ## Step 1: set the initial conditions and add to CSV file 
    gamma_variation_file = open('SIR_gamma_variation.csv', 'a')
    gamma_variation_file.write("{}, {}, {:.1f}\n".format( 1/beta, 1/gamma, beta/gamma ) )
    gamma_variation_file.close()

    S0 = N - I0 - R0 
    t = np.arange(NUM_DAYS)
    y0 = (S0, I0, R0)

    # Step 2: integrate the SIR equations over the time grid, t.

    y_array = odeint(deriv, y0, t) #ode returns columns and we need to convert to rows
    S, I, R = y_array.T #T will transpose our array (matrix)

    # Step 3: plot the data on three separate curves for S(t), I(t) and R(t)

    fig, ax = plt.subplots()
    ax.set_facecolor('gainsboro')

    ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/N, 'r', alpha=0.5, lw=3, label='Infectious')
    ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Removed')

    ax.set_xlabel('time(days)')
    ax.set_ylabel('fraction of population')

    #add legend
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    #add gridlines
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='w', lw=1, ls='-')

    #turn off bounding box
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    # make up the text box to display the parameters
    textstr = '\n'.join((
        'contact time = {} days'.format(1/beta), 'incubation time = {} days'.format(1/gamma),'R0 = {:.1f}'.format(beta/gamma) ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper right in axes coords

    ax.text(0.95, 0.8, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)

    # Save plot in directory folder with name of value varied and value
    plt.savefig('gamma_{:.0f}_days.png'.format(1/gamma))
    plt.close()