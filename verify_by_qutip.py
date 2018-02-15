#!/usr/bin/python3
import numpy as np
import h5py
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
test_pulse = './00000_spin_pi_speed_up.h5'

from quantum_optimal_control.helper_functions.qutip_verification import *
qutip_verification(test_pulse,atol=1e-3)
