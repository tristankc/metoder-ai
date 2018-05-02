from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))


def L_simple(w1,w2):
    sum_w12 = w1+w2

    return (logistic_z(w1)-1)**2 + logistic_z(w2)**2 + (logistic_z(sum_w12)-1)**2

def plotting():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    w1 = np.arange(-6, 6, 0.1)
    w2 = np.arange(-6, 6, 0.1)

    w1, w2 = np.meshgrid(w1, w2)

    Z = L_simple(w1,w2)

    # Plot the surface.
    surf = ax.plot_surface(w1, w2, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


#plotting()


def L_simple_derivated(w1, w2):
    w_1 = (2.0*np.exp(-w1)*(logistic_z(w1)-1.0))/((np.exp(-w1)+1)**2.0) + (2.0*np.exp(-w1-w2)*(logistic_z((w1+w2))-1.0))/((np.exp(-w1-w2)+1)**2)
    w_2 = (2.0*np.exp(-w2))/((np.exp(-w2)+1)**3.0) + (2.0*np.exp(-w1-w2)*(logistic_z((w1+w2))-1.0))/(np.exp(-w1-w2)+1)**2
    return [w_1, w_2]

def L_simple_derivated_w1(w1, w2):
    w_1 = (2.0*np.exp(-w1)*(logistic_z(w1)-1.0))/((np.exp(-w1)+1)**2.0) + (2.0*np.exp(-w1-w2)*(logistic_z((w1+w2))-1.0))/((np.exp(-w1-w2)+1)**2)
    return w_1

def L_simple_derivated_w2(w1, w2):
    w_2 = (2.0*np.exp(-w2))/((np.exp(-w2)+1)**3.0) + (2.0*np.exp(-w1-w2)*(logistic_z((w1+w2))-1.0))/(np.exp(-w1-w2)+1)**2
    return w_2

def gradient_descending(start_w1, start_w2, iterations, learningrate):

    w = [start_w1, start_w2]

    for x in range(iterations):

        w1 = w[0] - learningrate * L_simple_derivated_w1(w[0],w[1])
        w2 = w[1] - learningrate * L_simple_derivated_w2(w[0], w[1])

        if w1 > 6 or w1 < -6 or w2 > 6 or w2 < -6:
            return [w1, w2]

        w[0] = w1
        w[1] = w2

    return w

#print(gradient_descending(0.5,0.5,10000,0.1))



def plotting_with_points():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    w1 = np.arange(-6, 6, 0.1)
    w2 = np.arange(-6, 6, 0.1)

    w1, w2 = np.meshgrid(w1, w2)
    Z = L_simple(w1,w2)


    # Plot the surface.
    surf = ax.plot_surface(w1, w2, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    w1_points = []
    w2_points = []
    z_points = []

    w = gradient_descending(0.5, 0.5, 1000, 0.0001)
    w1_points.append(w[0])
    w2_points.append(w[1])
    z_points.append(L_simple(w[0], w[1]))

    a = gradient_descending(0.5, 0.5, 1000, 0.01)
    w1_points.append(a[0])
    w2_points.append(a[1])
    z_points.append(L_simple(a[0], a[1]))

    b = gradient_descending(0.5, 0.5, 1000, 0.1)
    w1_points.append(b[0])
    w2_points.append(b[1])
    z_points.append(L_simple(b[0], b[1]))

    c = gradient_descending(0.5, 0.5, 1000, 1)
    w1_points.append(c[0])
    w2_points.append(c[1])
    z_points.append(L_simple(c[0], c[1]))

    #d = gradient_descending(0.5, 0.5, 1000, 2)
    #w1_points.append(d[0])
    #w2_points.append(d[1])
    #z_points.append(L_simple(d[0], d[1]))

    e = gradient_descending(0.5, 0.5, 1000, 10)
    w1_points.append(e[0])
    w2_points.append(e[1])
    z_points.append(L_simple(e[0], e[1]))

    f = gradient_descending(0.5, 0.5, 1000, 100)
    w1_points.append(f[0])
    w2_points.append(f[1])
    z_points.append(L_simple(f[0], f[1]))

    for n in range(len(w1_points)):
        print("w1: " + str(round(w1_points[n],8)) + ". w2: " + str(round(w2_points[n],8)) + ". z: " + str(round(z_points[n],8)))


    plt.plot(w1_points,w2_points,z_points, 'go', markersize=4)


    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    plt.show()
#Task 1
#plotting()
#print(L_simple(6, -3))

plotting_with_points()







