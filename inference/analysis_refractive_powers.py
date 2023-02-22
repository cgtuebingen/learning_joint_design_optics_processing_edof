#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:13:55 2020

@author: Jieen Chen

This script plots the focal length relation between the two lenses for the
joint-design hybrid zooming. 

phi_slm    --- refractive power of the SLM
phi_l      --- refractive power of the lens
phi        --- refractive power of the whole system
m          --- magnification of the whole system
f_slm      --- focal length of the SLM
f_l        --- focal length of the lens
s_o_slm    --- distance between object and SLM
s_slm_l    --- distance between lens and SLM
s_l_sensor --- distance between lens and sensor

"""

import numpy as np
import matplotlib.pyplot as plt

f_slm = np.arange(-250,250,1)
s_o_slm = 2000
s_slm_l = 145
s_l_sensor = 100

phi_slm = 1/f_slm

phi_l = 1/s_l_sensor + (1+s_o_slm*phi_slm)/(s_slm_l*(1+s_o_slm*phi_slm)-s_o_slm)
#f_l = 100
#phi_l = 1/f_l
phi = (1-s_slm_l*phi_slm)/s_l_sensor + 1/(s_slm_l-s_o_slm*(1-s_slm_l*phi_slm))
m = s_l_sensor/(s_o_slm - s_slm_l*(1+s_o_slm * phi_slm))
f_l = 1/phi_l

# show plot of magnification vs focal length of the SLM
fig0, ax = plt.subplots()
ax.plot(f_slm, m, 'k-', label="Magnification")
plt.title("Magnification vs focal length of the SLM")
ax.legend()

#plt.show()
plt.savefig("magnification_vs_f_slm.png")

# show plot of the focal lengths of the lens and the SLM
fig1, ax = plt.subplots()
ax.plot(f_slm, f_l, 'r-', label="Focal length of the lens")
plt.title("Focal length of the lens vs focal length of the SLM")
ax.legend()

#plt.show()
plt.savefig("f_lens_vs_f_slm.png")