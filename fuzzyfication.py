#Importing Necessary Packages
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#Defining the Fuzzy Range from a speed of 30 to 90
x = np.arange(30, 80, 0.1)

#With the triangular membership functions
slow = fuzz.trimf(x, [30, 30, 50])
medium = fuzz.trimf(x, [30, 50, 70])
medium_fast = fuzz.trimf(x, [50, 60, 80])
full_speed = fuzz.trimf(x, [60, 80, 80])

#With the trapezoidal membership functions
slow = fuzz.trapmf(x, [20, 30, 40, 50])
medium = fuzz.trapmf(x, [30, 50, 60, 70])
medium_fast = fuzz.trapmf(x, [50, 60, 70, 80])
full_speed = fuzz.trapmf(x, [60, 80, 90, 100])

#With the gaussian membership functions
full_speed = fuzz.gaussmf(x, 80, 4)
medium_fast = fuzz.gaussmf(x, 60, 4)
medium = fuzz.gaussmf(x, 50, 4)
slow = fuzz.gaussmf(x, 30, 4)

#With the generalized bell membership functions
full_speed = fuzz.gbellmf(x, 8,4,80)
medium_fast = fuzz.gbellmf(x, 8,4,60)
medium = fuzz.gbellmf(x, 8,4,50)
slow = fuzz.gbellmf(x, 8,4,30)

#With the sigmoidal membership functions
full_speed = fuzz.sigmf(x, 80,2)
medium_fast = fuzz.sigmf(x, 60,2)
medium = fuzz.sigmf(x, 50,2)
slow = fuzz.sigmf(x, 30,2)

#With the z-shaped membership functions
full_speed = fuzz.smf(x, 60,80)
medium_fast = fuzz.smf(x, 50,60)
medium = fuzz.smf(x, 30,50)
slow = fuzz.smf(x, 20,30)

#With the s-shaped membership functions
full_speed = fuzz.zmf(x, 60,80)
medium_fast = fuzz.zmf(x, 50,60)
medium = fuzz.zmf(x, 30,50)
slow = fuzz.zmf(x, 20,30)

#With the pi-shaped membership functions
full_speed = fuzz.pimf(x, 60,70,80,100)
medium_fast = fuzz.pimf(x, 50,55,60,80)
medium = fuzz.pimf(x, 30,45,50,60)
slow = fuzz.pimf(x, 20,25,35,50)

#Plotting the Membership Functions Defined
plt.figure()
plt.plot(x, full_speed, 'b', linewidth=1.5, label='Full Speed')
plt.plot(x, medium_fast, 'k', linewidth=1.5, label='Medium Fast')
plt.plot(x, medium, 'm', linewidth=1.5, label='Medium Powered')
plt.plot(x, slow, 'r', linewidth=1.5, label='Slow')
plt.title('Penalty Kick Fuzzy')
plt.ylabel('Membership')
plt.xlabel("Speed (Miles Per Hour)")
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),ncol=1, fancybox=True, shadow=True)
plt.show()