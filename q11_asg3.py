import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt

def f1(x):
  if(x>1):
    return 0
  elif(x<-1):
    return 0
  else:
    return 1
    
def f2(x):
  if(x>1):
    return 0
  elif(x<-1):
    return 0
  else:
    return 1
"""
  For convolution of some other functions change f1  and f2 only.
"""

n=256
data1=np.zeros(n,dtype=np.complex_)
data2=np.zeros(n,dtype=np.complex_)
x=np.zeros(n,dtype=np.complex_)
xmin=-7
xmax=7
dx=(xmax-xmin)/(n-1)
for i in range(n):
  x[i]=xmin + i*dx
  data1[i]=f1(x[i])
  data2[i]=f2(x[i])
  

dft1=np.fft.fft(data1,norm='ortho')
dft2=np.fft.fft(data2,norm='ortho')
dft=dft1*dft2


conv=np.fft.ifftshift(np.fft.ifft(dft,norm='ortho'))
conv=conv*dx*np.sqrt(n)


plt.plot(x.real,conv.real,".",label="Convolution of Box function with itself")
plt.plot(x.real,data1,color="red",label="Box function")
plt.legend()
plt.ylabel("$f(x)$")
plt.show()
