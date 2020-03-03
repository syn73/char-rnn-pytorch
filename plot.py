import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
args = argparser.parse_args()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#fig.subplots_adjust(left=0.02,right=0.99,bottom=0.04,top=0.99)

def animate(i):
    data = np.genfromtxt(args.filename, delimiter=',', skip_header=1,
        names=['steps','loss','smooth_loss','time_elapsed'])
    ax1.clear()
    ax1.plot(data['steps'], data['loss'])
    ax1.plot(data['steps'], data['smooth_loss'])

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
