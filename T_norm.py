import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


#Defining the T Minimum Norm Function
def t_min_norm(mfx,mfy):
    tnorm = np.fmin(mfx, mfy)
    return tnorm

#Defining the T Product Norm Function
def t_prod_norm(mfx,mfy):
    tnorm = mfx * mfy
    return tnorm

#Defining the T Lukasiewicz Norm Function
def t_luk_norm(mfx,mfy):
    tnorm = np.fmax(mfx + mfy-1, 0)
    return tnorm


if __name__=="__main__":

    x = np.arange(0, 150, 0.1)

    #Defining sigmoidal membership function
    full_speed = fuzz.sigmf(x, 50,2)
    slow = fuzz.sigmf(x, 30,2)


    #Finding the Intersection
    min_norm = t_min_norm(full_speed,slow)
    prod_norm = t_prod_norm(full_speed,slow)
    luk_norm = t_luk_norm(full_speed,slow)

    #Plotting the Membership Functions Defined
    plt.figure()
    plt.plot(x, full_speed, 'b', linewidth=1.5, label='Full Speed')
    plt.plot(x, slow, 'r', linewidth=1.5, label='Slow')
    plt.plot(x, min_norm, '--', linewidth=1.5, label='Min T-norm')
    plt.plot(x, prod_norm, '--', linewidth=1.5, label='Prod T-norm')
    plt.plot(x, luk_norm, '--', linewidth=1.5, label='Luk T-norm')
    plt.ylabel('Membership')
    plt.xlabel("Speed (Miles Per Hour)")
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),ncol=1, fancybox=True, shadow=True)
    plt.show()
