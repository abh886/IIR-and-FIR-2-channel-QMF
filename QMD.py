# -*- coding: utf-8 -*-
"""
To design a 2 channel FIR Quadrature Mirror Filter Bank
"""
from numpy import flip,fliplr,linspace,abs,convolve
from numpy import pi,cos,sin,log10
from scipy.signal import qmf,freqz,hann,upfirdn
import matplotlib.pyplot as plt
from scipy.fftpack import fft



n=linspace(0,26)
ts=cos(0.5*pi*n)


b=hann(26)
C=hann(26)
c=qmf(b)
D=qmf(C)

#q1=convolve(ts,b)
#q2=convolve(ts,c)

q11=upfirdn(ts,b,up=1,down=2)
q22=upfirdn(ts,c,up=1,down=2)

q111=upfirdn(q11,[1],up=2,down=1)
q222=upfirdn(q22,[1],up=2,down=1)

#Q1=convolve(q111,C)
#Q2=convolve(q222,D)

Q=q111+q222

w0,H0=freqz(b,1,256)
w1,H1=freqz(c,1,256)

D=fft(b,256)
E=fft(c,256)
F=20*log10(abs(D))
G=20*log10(abs(E))

H=20*log10(abs(H0))
I=20*log10(abs(H1))

plt.stem(n,ts)
plt.title('input signal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()

plt.stem(b)
plt.title('FIR lowpass filter')
plt.xlabel('n')
plt.ylabel('Amplitude')
#plt.grid()
plt.show()

plt.stem(c)
plt.title('FIR Highpass filter')
plt.xlabel('n')
plt.ylabel('Amplitude')
#plt.grid()
plt.show()

plt.plot(w0/(2*pi),H,'r')
plt.title('Low Pass filter frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

plt.plot(w1/(2*pi),I,'g')
plt.title('High pass filter frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('amplitude')
plt.grid()
plt.show()

plt.plot(w1/(2*pi),G,'b')
plt.title('Analysis Filter bank frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

plt.plot(w1/(2*pi),G,'r')
plt.title('Synthesis Filter bank frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

plt.stem(Q)
plt.title('Compressed signal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()