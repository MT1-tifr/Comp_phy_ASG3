import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def gaus(x,y):
  return(np.exp(-(x*x)-(y*y)))
  
def fftgaus(x,y):
  return(np.exp((-(x*x)-(y*y))/4)/2)

 
n=100
ymax=xmax=20
ymin=xmin=-20
dx=(xmax-xmin)/(n-1)
dy=(ymax-ymin)/(n-1)
x= np.arange(xmin,xmax+dx,dx,dtype=np.complex_)
y= np.arange(ymin,ymax+dy,dy,dtype=np.complex_)



data=np.zeros([n,n],dtype=np.complex_)

kx=np.fft.fftfreq(n,dx)
ky=np.fft.fftfreq(n,dy)
kx=2*np.pi*kx
ky=2*np.pi*ky

for i in range(0,n,1):
  for j in range(0,n,1):
    data[i][j]=gaus(x[i],y[j])
    
    
nft=np.fft.fft2(data,norm='ortho')
print(nft.size)

aft=np.zeros([n,n],dtype=np.complex_)

for i in range(0,n,1):
  for j in range(0,n,1):
    aft[i][j]=dx*dy*(n/(2.0*np.pi))*(np.exp(-1j*kx[i]*xmin + -1j*ky[j]*ymin ))*nft[i][j]

fig = plt.figure()
ax = plt.axes(projection="3d")

x=np.zeros(n*n)
y=np.zeros(n*n)
z=np.zeros(n*n)
zz=np.zeros(n*n)
for i in range(0,n,1):
  for j in range(0,n,1):
    x[j+i*n]=kx[i]
    y[j+i*n]=ky[j]
    z[j+i*n]=aft[i][j].real
    zz[j+i*n]=fftgaus(kx[i],kx[j])

   
   
#print(y)
ax.plot3D(x, y, z, '.r',label="Fourier Transform of Gaussian using numpy.fft.fft2")
ax.plot3D(x, y, zz, '.k',label="Analytic Fourier Transform of Gaussian")
plt.legend()
plt.show()


#plt.plot(k,aft.real,".",label="Fourier Transform using Numpy")
#plt.plot(kk,box.real,"",label="Analytic Fourier Transform(Box function)")
#plt.legend(fontsize=15)
#plt.title("Fourier transform of $sinc(x)$",fontsize=17)
#plt.xlabel("k",fontsize=15)
#plt.ylabel("F(k)",fontsize=15)
#plt.show()
