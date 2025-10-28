import numpy as np
from numba import njit

@njit(cache=True)
def solve(f, h, t0, tf, r0, t_points, r_points, adaptive, max_tol, *args):
    '''
    Doc string for solve function
    '''

    t = t0
    i = 0

    r_points[0] = r0
    t_points[0] = t0

    while t < tf:
        r_i = r_points[i]

        if adaptive:
            r1 = numba_rk4(f, t, r_i, h, *args)

            r_half = numba_rk4(f, t, r_i, h / 2, *args)

            r2 = numba_rk4(f, t + h / 2, r_half, h / 2, *args)

            Err = np.max(np.abs(r2 - r1))
            Err = max(Err, 1e-14)

            h = h*(max_tol / Err)**(1/5)

        h = min(h, tf - t)  # Do not step beyond tf/avoid overshooting
        r = numba_rk4(f, t, r_i, h, *args)
        r_points[i + 1] = r
        t += h
        t_points[i + 1]= t
        i += 1

    return t_points[:i+1], r_points[:i+1]

@njit(cache=True)
def numba_rk4(f, t, r, h, *args):
    '''
    Runge-Kutta fourth order (RK4) using Numba
    '''
    k1 = h * f(r, t, *args)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h, *args)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h, *args)
    k4 = h * f(r + k3, t + h, *args)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6


'''
def solve(f, h, r0, t_points):

        #Runge-Kutta fourth order (RK4) using Numba

        n = len(t_points)
        m = len(r0)
        points = np.zeros((n,m))
        r = r0.copy()

        for i, t in enumerate(t_points):
            points[i, :] = r
            k1 = h * f(r, t)
            k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
            k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
            k4 = h * f(r + k3, t + h)
            r += (k1 + 2 * (k2 + k3) + k4) / 6
        return t_points, points

'''