import skfuzzy as sk
import numpy as np
import matplotlib.pyplot as plt

#Defining the Numpy array for Tip Quality
x_qual = np.arange(0, 11, 1)
#Defining the Numpy array for two membership functions (Triangular)
qual_lo = sk.trimf(x_qual, [0, 0, 5])
qual_md = sk.trimf(x_qual, [0, 5, 10])

# UNION: Finding the Maximum (Fuzzy Or)
or_operation1 = sk.fuzzy_or(x_qual,qual_lo,x_qual,qual_md)

plt.plot(or_operation1[0], or_operation1[1])
plt.show()

# INSERTECTION: Finding the Minimum (Fuzzy AND)
or_operation2 = sk.fuzzy_and(x_qual,qual_lo,x_qual,qual_md)

plt.plot(or_operation2[0], or_operation2[1])
plt.show()

# COMPLEMENT: Finding the Complement (Fuzzy NOT)
or_operation3 = sk.fuzzy_not(qual_lo)

plt.plot(x_qual, qual_lo)
plt.plot(x_qual, or_operation3)
plt.show()

# PRODUCT: Finding the Product (Fuzzy Cartesian)
or_operation4 = sk.cartprod(qual_lo, qual_md)

print("PRODUCT")
print(or_operation4)

# DIFFERENCE: Finding the Difference (Fuzzy Subtract)
or_operation5 = sk.fuzzy_sub(x_qual,qual_lo,x_qual,qual_md)

plt.plot(or_operation5[0], or_operation5[1])
plt.show()

