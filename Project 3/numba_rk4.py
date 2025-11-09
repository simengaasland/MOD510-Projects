import numpy as np
from numba import njit

@njit
def solve(f, h, t0, tf, r0, t_points, r_points, adaptive, max_tol, *args):
    '''
    Solves a system of ordinary differential equations (ODEs) using the
    Runge-Kutta fourth-order (RK4) method, optionally with adaptive step size.

    Parameters
    ----------
    f : function
        Function that returns the derivative dr/dt = f(r, t, *args).
    h : float
        Initial time step size.
    t0 : float
        Start time of the simulation.
    tf : float
        End time of the simulation.
    r0 : array
        Initial condition(s).
    t_points : array
        Array for storing time points.
    r_points : array
        Array for storing solution values corresponding to t_points.
    adaptive : bool
        If True, enables adaptive step sizing.
    max_tol : float
        Maximum local error tolerance for adaptive step.
    *args : tuple
        Additional arguments passed to the derivative function f.

    Returns
    -------
    t_points : array
            Array of time points.
        r_points: array
            Array of solution values corresponding to t_points.
    '''

    t = t0 #Initial time
    i = 0 #Step counter

    #Initialize
    r_points[0] = r0
    t_points[0] = t0

    #Time stepping loop
    while t < tf:
        r_i = r_points[i]

        if adaptive:
            #Full step
            r1 = numba_rk4(f, t, r_i, h, *args)

            #First half step
            r_half = numba_rk4(f, t, r_i, h / 2, *args)

            #Second half step
            r2 = numba_rk4(f, t + h / 2, r_half, h / 2, *args)

            #Estimate error and adjust step size
            Err = np.max(np.abs(r2 - r1))
            Err = max(Err, 1e-14) #Avoid divition by zero

            h = h*(max_tol / Err)**(1/5)

        #Avoid stepping beyond tf
        h = min(h, tf - t)

        #Advance one step
        r = numba_rk4(f, t, r_i, h, *args)
        r_points[i + 1] = r
        t += h
        t_points[i + 1]= t
        i += 1

    #Trim and return unused preallocated arrays
    return t_points[:i+1], r_points[:i+1]

@njit
def numba_rk4(f, t, r, h, *args):
    '''
    Runge-Kutta fourth order (RK4) using Numba
    '''
    k1 = h * f(r, t, *args)
    k2 = h * f(r + 0.5 * k1, t + 0.5 * h, *args)
    k3 = h * f(r + 0.5 * k2, t + 0.5 * h, *args)
    k4 = h * f(r + k3, t + h, *args)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
