import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as sk


#Defining the Numpy array for Tip Quality
x_qual = np.arange(0, 11, 1)

#Defining the Numpy array for Triangular membership functions
qual_lo1 = sk.trimf(x_qual, [0, 0, 5])
plt.plot(x_qual, qual_lo1)
plt.show()

#Defining the Numpy array for Trapezoidal membership functions
qual_lo2 = sk.trapmf(x_qual, [0, 0, 5,5])
plt.plot(x_qual, qual_lo2)
plt.show()

#Defining the Numpy array for Gaussian membership functions
qual_lo3 = sk.gaussmf(x_qual, np.mean(x_qual), np.std(x_qual))
plt.plot(x_qual, qual_lo3)
plt.show()


#Defining the Numpy array for Generalized Bell membership functions
qual_lo4 = sk.gbellmf(x_qual, 0.5, 0.5, 0.5)
plt.plot(x_qual, qual_lo4)
plt.show()

qual_lo5 = sk.sigmf(x_qual, 0.5,0.5)
plt.plot(x_qual, qual_lo5)
plt.show()
