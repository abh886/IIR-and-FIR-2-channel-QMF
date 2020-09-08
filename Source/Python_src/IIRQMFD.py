# -*- coding: utf-8 -*-
"""
To pass a signal through a 2 channel IIR Quadrature mirror  Filter bank
"""

from numpy import flip, fliplr, linspace, abs, convolve
from numpy import pi, cos, sin, log10
from scipy.signal import qmf, freqz, resample_poly
import matplotlib.pyplot as plt
from scipy.fftpack import fft

n = linspace(0, 5)
ts = cos(0.5 * pi * n)

C = [.1, .2, .3, .1]
b = [.1, .2, .3, .1]
c = qmf(b)
w0, H0 = freqz(b, 1, 256)
w1, H1 = freqz(c, 1, 256)

D = qmf(C)
w2, H2 = freqz(C, 1, 256)
w3, H3 = freqz(D, 1, 256)

q1 = convolve(ts, b)
q2 = convolve(ts, c)

q11 = resample_poly(q1, up=1, down=2)
q22 = resample_poly(q2, up=1, down=2)

Q1 = convolve(q11, C)
Q2 = convolve(q22, D)

q1111 = resample_poly(q11, up=2, down=1)
q2222 = resample_poly(q22, up=2, down=1)

Q = q1111 + q2222

D = fft(b, 256)
E = fft(c, 256)
F = 20 * log10(abs(D))
G = 20 * log10(abs(E))

H = 20 * log10(abs(H0))
I = 20 * log10(abs(H1))
J = 20 * log10(abs(H2))
K = 20 * log10(abs(H3))

plt.stem(b, use_line_collection=True)
plt.title('IIR lowpass filter')
plt.xlabel('n')
plt.ylabel('Amplitude')
# plt.grid()
plt.show()

plt.stem(c)
plt.title('IIR Highpass filter')
plt.xlabel('n')
plt.ylabel('Amplitude')
# plt.grid()
plt.show()

plt.plot(w0 / (2 * pi), H, 'r')
plt.title('Low Pass filter frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

plt.plot(w1 / (2 * pi), I, 'g')
plt.title('High pass filter frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('amplitude')
plt.grid()
plt.show()

plt.plot(w1 / (2 * pi), I, 'b', w0 / (2 * pi), H, 'g')
plt.title('Analysis Filter bank frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

plt.plot(w2 / (2 * pi), J, 'k', w3 / (2 * pi), K, 'r')
plt.title('Synthesis  Filter bank frequency response')
plt.xlabel('Normalized frequency')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

plt.stem(ts)
plt.title('input signal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()

plt.stem(Q)
plt.title('Reconstructed signal')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()
