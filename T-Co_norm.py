import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


#Defining the Maximum T Co Norm Function
def tco_max_norm(mfx,mfy):
    tnorm = np.fmax(mfx, mfy)
    return tnorm

#Defining the Probabilistic T Co Sum Norm Function
def tco_probsum_norm(mfx,mfy):
    tnorm = mfx + mfy - mfx * mfy
    return tnorm

#Defining the Lukasiewicz T Co Norm Function
def tco_luk_norm(mfx,mfy):
    tnorm = np.fmin(mfx + mfy, 1)
    return tnorm


if __name__=="__main__":

    x = np.arange(0, 110, 0.1)

    #Defining sigmoidal membership function
    full_speed = fuzz.sigmf(x, 80,2)
    slow = fuzz.sigmf(x, 30,2)


    #Finding the Intersection
    max_norm = tco_max_norm(full_speed,slow)
    probsum_norm = tco_probsum_norm(full_speed,slow)
    luk_norm = tco_luk_norm(full_speed,slow)

    #Plotting the Membership Functions Defined
    plt.figure()
    plt.plot(x, full_speed, 'b', linewidth=1.5, label='Full Speed')
    plt.plot(x, slow, 'r', linewidth=1.5, label='Slow')
    plt.plot(x, max_norm, '--', linewidth=1.5, label='Max Tco-norm')
    plt.plot(x, probsum_norm, '--', linewidth=1.5, label='Probsum Tco-norm')
    plt.plot(x, luk_norm, '--', linewidth=1.5, label='Luk Tco -norm')
    plt.ylabel('Membership')
    plt.xlabel("Speed (Miles Per Hour)")
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),ncol=1, fancybox=True, shadow=True)
    plt.show()
