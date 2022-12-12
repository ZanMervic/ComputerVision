from matplotlib import pyplot as plt
import numpy as np


#Task B --------------------------------------------------------------------------------------------------------------

#Calculate the disparity for a range of distances pz
def task_b(f = 2.5e-3, T = 0.12):
    pz = np.linspace(0.01, 2, 400)
    disparities = f*T/pz
    plt.plot(pz, disparities)
    plt.xlabel('Distance (m)')
    plt.ylabel('Disparity (m)') #TODO: check if this is correct
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------

# task_b()