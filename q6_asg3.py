import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def f(x):
  return 10
  
def delta(x):
  if x==0:
   return 16000 
  else:
   return 0

 
n=1024
xmax=100
xmin=-100
dx=(xmax-xmin)/(n-1)
x= np.arange(xmin,xmax+dx,dx,dtype=np.complex_)



data=np.zeros(n,dtype=np.complex_)
analytic=np.zeros(n,dtype=np.complex_)
k=np.fft.fftfreq(n,dx)
k=2*np.pi*k
kk=np.linspace(k.min(),k.max(),num=n,endpoint=True)
for i in range(0,n,1):
  data[i]=f(x[i])
  analytic[i]=delta(kk[i])


plt.subplot(121)
plt.plot(x,data.real,label="Constant Function")
plt.legend()
plt.title("Constant Function")
plt.xlabel("x")
plt.ylabel("F(x)")
nft=np.fft.fft(data,norm='ortho')


aft=dx*np.sqrt(n/(2.0*np.pi))*(np.exp(-1j*k*x.min()))*nft

plt.subplot(122)
plt.plot(k,aft.real,"o",label="Fourier Transform of f(x)=10 using Numpy")
plt.plot(kk,analytic.real,"-",label="Analytic Fourier Transform of f(x)=10")
plt.legend()
plt.title("Fourier transform of constant function f(x)=10")
plt.xlabel("k")
plt.ylabel("F(k)")
axes=plt.gca()
axes.set_ylim([0,10000])
plt.show()
