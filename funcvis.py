from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)

sig = 1/(1 + np.exp(-x))
tanh = np.tanh(x)

custom = sig * tanh


fig, ax = plt.subplots()
ax.hlines(0, -10, 10, color='black', linestyle='--', label='__nolegend__')
ax.hlines(1, -10, 10, color='black', linestyle='--', label='__nolegend__')
ax.hlines(-1, -10, 10, color='black', linestyle='--', label='__nolegend__')
ax.plot(x, sig, color='blue', linewidth=1, label = 'sigmoid')
ax.plot(x, tanh, color='red', linewidth=1, label = 'tanh')
ax.plot(x, custom, color='green', linewidth=3, label = 'custom')
ax.legend(loc='upper left')

plt.show()