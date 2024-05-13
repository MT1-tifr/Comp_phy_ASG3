import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt

def f1(x):
  return np.exp(-x**2)
    
def f2(x):
    return np.exp(-4*x**2)

def analytical(x):
    return (np.sqrt(np.pi/5)*np.exp(-4*x**2 / 5))

n=1024
data1=np.zeros(n,dtype=np.complex_)
data2=np.zeros(n,dtype=np.complex_)
truesol=np.zeros(n,dtype=np.complex_)
x=np.zeros(n,dtype=np.complex_)
xmin=-10
xmax=10
dx=(xmax-xmin)/(n-1)
for i in range(n):
  x[i]=xmin + i*dx
  data1[i]=f1(x[i])
  data2[i]=f2(x[i])
  truesol[i]= analytical(x[i])

dft1=np.fft.fft(data1,norm='ortho')
dft2=np.fft.fft(data2,norm='ortho')
dft=dft1*dft2


conv=np.fft.ifftshift(np.fft.ifft(dft,norm='ortho'))
conv=conv*dx*np.sqrt(n)


plt.plot(x.real,conv.real,".",label="Convolution of h(x) and g(x)")
plt.plot(x.real,data1,color="red",label="h(x)=exp(-x^2)")
plt.plot(x.real,data2,color="green",label="g(x)=exp(-4x^2)")
plt.plot(x.real,truesol, "--", color="black",label="Analytical Convolution")
plt.legend()
plt.ylabel("$f(x)$")
plt.show()
