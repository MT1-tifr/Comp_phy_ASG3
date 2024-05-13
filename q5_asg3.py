import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import timeit

size_range=np.arange(4,100,2)
t_dft=np.zeros(size_range.size)
t_dft_np=np.zeros(size_range.size)
N=1000
counter=0

stmt1="""

for i in range(n):
  for m in range(n):
    dft_data[i]+=data[m]*np.exp((-1j*m*i*2*np.pi)/n)
  dft_data[i]=dft_data[i]/np.sqrt(n)
   

"""
stmt2="""

np_dft_data=np.fft.fft(data,norm='ortho')

"""
for j in size_range:
  my_setup="""
import numpy as np
n=%d
data=np.zeros(n,dtype=np.complex_)
for i in range(n):
  data[i]=i

dft_data=np.zeros(n,dtype=np.complex_)
np_dft_data=np.zeros(n,dtype=np.complex_)

"""%(j)
  t_dft[counter]=timeit.timeit(setup=my_setup,stmt=stmt1,number=N)
  t_dft_np[counter]=timeit.timeit(setup=my_setup,stmt=stmt2,number=N)
  counter+=1
  
t_dft=t_dft/N
t_dft_np=t_dft_np/N
plt.plot(size_range,t_dft,".-",label="DFT")
plt.plot(size_range,t_dft_np,".-",label="DFT numpy")
plt.xlabel("N")
plt.ylabel("Time")
plt.legend()
plt.show()
