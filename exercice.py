#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
from numpy import linalg as LA
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from random import random
from scipy.integrate import quad




# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    angle = np.arctan(cartesian_coordinates[0]/cartesian_coordinates[1])
    rayon = LA.norm(cartesian_coordinates)
    return (rayon, angle)


def find_closest_index(values: np.ndarray, number: float) -> int:
    result =  np.absolute(values - number)
    index = result.argmin()
    return index

def fonction():
    def f(x):
        return ((x**2) * np.sin(1 / (x**2))) + x
    xp = np.linspace(-1, 1, 250)
    yp = f(xp)
    p1 = interp.interp1d(xp, yp)
    plt.plot(xp,yp)
    plt.show()

def montCarlos():
    xi= []
    yi= []
    xo= []
    yo= []
    nbPoints = 5000

    for i in range(nbPoints):
            x,y = np.random.random(),np.random.random()
            if (x**2)+ (y**2) < 1: # carré de la distance OM
                xi.append(x)
                yi.append(y)
            else:
                xo.append(x)
                yo.append(y)

    plt.plot(np.array(xi), np.array(yi), 'o')
    plt.plot(np.array(xo), np.array(yo), 'o')
    plt.title("Calcul de pi par la methode Monte Carlo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def integration():
    def f(x):
        return np.e ** (-(x**2))
    Ih, err = quad(f, -np.inf, np.inf)



    xp = np.linspace(-4, 4)
    yp = f(xp)
    p1 = interp.interp1d(xp, yp)
    plt.plot(xp,yp)
    plt.fill_between(xp, yp, color='blue', alpha=0.3)
    plt.show()

    print("Integrale =", Ih, " Erreur =", err)




        


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())
    print(coordinate_conversion(np.array([1,1])))
    print(find_closest_index(np.array([1,4,6,8,9]), 9))
    fonction()
    montCarlos()
    integration()



    pass
