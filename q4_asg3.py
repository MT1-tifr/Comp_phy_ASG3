import matplotlib.pyplot as plt
import numpy as np
def analytical(x):
    return (np.exp(-x**2 / 4)/np.sqrt(2))
# Open the file for reading
with open('fftgauss.txt', 'r') as file:
    # Initialize empty lists to store the data
    x = []
    y = []

    # Read each line in the file
    for line in file:
        # Split the line into two parts based on whitespace
        parts = line.split()

        # Convert the parts to floating point numbers and append them to the x and y lists
        x.append(float(parts[0]))
        y.append(float(parts[1]))
y=np.fft.fftshift(y)
x=np.fft.fftshift(x)
n=1024
truesol=np.zeros(n,dtype=np.complex_)
x1=np.zeros(n,dtype=np.complex_)
x1min=-10
x1max=10

dx1=(x1max-x1min)/(n-1)
for i in range(n):
  x1[i]=x1min + i*dx1
  truesol[i]= analytical(x1[i])
# Plot the data
plt.plot(x, y, label="Fourier Transform using FFTW")
plt.plot(x1, truesol, linestyle='--', label="Analytical Fourier Transform")
plt.xlabel('k')
plt.ylabel('F(k)')
plt.title('Fourier Transform of Gaussian Function exp(-x^2)')
plt.legend()
plt.grid(True)
plt.show()
