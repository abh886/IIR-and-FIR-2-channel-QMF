# -*- coding: utf-8 -*-
"""
To design a 2 channel FIR Quadrature Mirror Filter Bank
"""
from numpy import flip,fliplr,linspace,abs,convolve
from numpy import pi,cos,sin,log10
from scipy.signal import qmf,freqz,hann,hamming
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import resample_poly
from scipy.signal import firwin,lfilter

fpa = 1000
fsa = 3000
fc = (fpa+fsa)/2
n=linspace(0,50)
ts=cos(2*pi*400*n/5000)

b=hamming(26)
C=hamming(26)
c=qmf(b)
D=qmf(C)

f1 = firwin(30,fc/(5000/2),window = 'hamming')
op = lfilter(f1,1,ts)
op1 = qmf(op)

#QMF procedure
q1=convolve(ts,op)
q2=convolve(ts,op1)

q11=resample_poly(ts,up=1,down=2)
q22=resample_poly(ts,up=1,down=2)

q111=resample_poly(q11,up=2,down=1)
q222=resample_poly(q22,up=2,down=1)

Q1=convolve(q111,op)
Q2=convolve(q222,op1)

Q=q111+q222

w0,H0=freqz(b,1,256)
w1,H1=freqz(c,1,256)


D=fft(b,256)
E=fft(c,256)
F=20*log10(abs(D))
G=20*log10(abs(E))

H=20*log10(abs(H0))
I=20*log10(abs(H1))

plt.stem(n,ts,use_line_collection=True)
plt.title('input signal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()

plt.stem(b,use_line_collection = True)
plt.title('FIR lowpass filter')
plt.xlabel('n')
plt.ylabel('Amplitude')
#plt.grid()
plt.show()

plt.stem(c,use_line_collection = True)
plt.title('FIR Highpass filter')
plt.xlabel('n')
plt.ylabel('Amplitude')
#plt.grid()
plt.show()

plt.plot(w0/(2*pi),G,'b')
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

plt.stem(op,use_line_collection = True)
plt.title('output response')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()

plt.stem(Q,use_line_collection = True)
plt.title('Reconstructed signal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()