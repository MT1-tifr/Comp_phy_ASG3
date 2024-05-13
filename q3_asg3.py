import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def sinc(x):
  if(x==0):
   return 1
  else:
   return np.sin(x)/x
  
def Box(x):
  if(x>1):
    return 0
  elif(x<-1):
    return 0
  else:
    return(np.sqrt(np.pi/2))
 
n=1024   
xmax=100
xmin=-100
dx=(xmax-xmin)/(n-1)
x= np.arange(xmin,xmax+dx,dx,dtype=np.complex_)

data=np.zeros(n,dtype=np.complex_)
box=np.zeros(n,dtype=np.complex_)
k=np.fft.fftfreq(n,dx)
k=2*np.pi*k
kk=np.linspace(k.min(),k.max(),num=n,endpoint=True)
for i in range(0,n,1):
  data[i]=sinc(x[i])
  box[i]=Box(kk[i])

# Open the file for reading
with open('fftdata.txt', 'r') as file:
    # Initialize empty lists to store the data
    xc = []
    yc = []

    # Read each line in the file
    for line in file:
        # Split the line into two parts based on whitespace
        parts = line.split()

        # Convert the parts to floating point numbers and append them to the x and y lists
        xc.append(float(parts[0]))
        yc.append(float(parts[1]))
        
# Open the file for reading
with open('fftgsl.txt', 'r') as file:
    # Initialize empty lists to store the data
    xgsl = []
    ygsl = []

    # Read each line in the file
    for line in file:
        # Split the line into two parts based on whitespace
        parts = line.split()

        # Convert the parts to floating point numbers and append them to the x and y lists
        xgsl.append(float(parts[0]))
        ygsl.append(float(parts[1]))
plt.subplot(121)
plt.plot(x,data.real,label="Sinc Function")
plt.legend()
plt.title("Sinc Function")

nft=np.fft.fft(data,norm='ortho')
aft=dx*np.sqrt(n/(2.0*np.pi))*(np.exp(-1j*k*x.min()))*nft

plt.subplot(122)
plt.plot(k,aft.real,".", label="Fourier Transform using Numpy")
plt.plot(kk,box.real,color="red", label="Analytic Fourier Transform")
plt.plot(xc,yc,".",color='green', label="FFTW using C")
plt.plot(xgsl,ygsl,".",color='black', label="Fourier Transform using GSL")
plt.title("Fourier transform")
plt.xlabel("k")
plt.ylabel("F(k)")
plt.legend()
plt.show()


