import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(0.0,0.5,size=(5000,10))**2
a = a/np.sum(a,axis=1)[:,None]  # Normalize

plt.pcolor(a)