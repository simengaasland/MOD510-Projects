import numpy as np
import math
import numba_rk4

#Remove Numba warnings to keep notebook clean
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning
warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

class ODESolver:
    '''
    This class solves general ordinary differential equations (ODEs)
    and also systems of ODES using:
        - Euler's method
        - Runge-Kutta 2nd order (RK2)
        - Runge-Kutta 4th order (RK4)
        - A faster RK4 implementation (via numba_rk4)
    Supports both fixed and adaptive step size.
    '''
    def __init__(self, f, t0, tf, h, r0, method = 'rk4', max_tol = 1e-5, adaptive = True, args = ()):
        '''
        Initialize the ODE solver.

        Parameters
        ----------
        f : callable
            Function that defines the ODE system: f(r, t, *args)
        t0 : float
            Initial time
        tf : float
            Final time
        h : float
            Step size
        r0 : list
            Initial condition(s)
        method : str, optional
            Integration method ('euler', 'rk2', 'rk4', 'fast_rk4')
        max_tol : float, optional
            Maximum allowed local error for adaptive step size
        adaptive : bool, optional
            If True, use adaptive step size
        args : tuple, optional
            Extra arguments passed to f
        '''

        #Store parameters
        self.f = f
        self.t0 = t0
        self.tf = tf
        self.h = h
        if adaptive:
            #Allocating a large number  of n's
            self.n = int(1e6)
        else:
            self.n = math.ceil((tf - t0) / h)

        #Allocating for time and solution
        self.t_points = np.zeros(self.n + 2, float)
        self.r0 = np.array(r0, float)

        #Select method from dictionary
        self.method = method
        self.choosen_method = {'euler': self.__euler,
                               'rk2': self.__rk2,
                               'rk4': self.__rk4,
                               'fast_rk4': numba_rk4.solve
                               }[method]

        self.r_points = np.zeros((self.n + 2, len(self.r0)), float)
        self.adaptive = adaptive

        #Selecting step exponent for method
        self.step_exponent = {"euler":1/2, "impeuler":1/3, "rk2":1/3, "rk4":1/5, 'fast_rk4': None}[method]

        self.max_tol = max_tol
        self.args = args

    def __euler(self, t, r, h):
        '''
        Eulers method
        '''
        return r + h * self.f(r, t, *self.args)

    def __rk2(self, t, r, h):
        '''
        Runge-Kutta second order (RK2)
        '''
        k1 = self.f(r, t, *self.args)
        k2 = self.f(r + 0.5 * k1 * h, t + 0.5 * h, *self.args)
        return r + h * k2

    def __rk4(self, t, r, h):
        '''
        Runge-Kutta fourth order (RK4)
        '''
        k1 = h * self.f(r, t, *self.args)
        k2 = h * self.f(r + 0.5 * k1, t + 0.5 * h, *self.args)
        k3 = h * self.f(r + 0.5 * k2, t + 0.5 * h, *self.args)
        k4 = h * self.f(r + k3, t + h, *self.args)
        return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self):
        '''
        Solves ODEs using users method of choice

        Returns
        ----------
        t_points : array
            Array of time points
        r_points: array
            Array of solution values corresponding to t_points
        '''

        #Initialize
        self.r_points[0] = self.r0
        self.t_points[0] = self.t0

        #If fast RK4 is chosen
        if self.choosen_method == numba_rk4.solve:
            self.t_points, self.r_points = self.choosen_method(
                                            self.f, self.h, self.t0, self.tf,
                                            self.r0, self.t_points, self.r_points,
                                            self.adaptive, self.max_tol, *self.args)
        #If a non Numba method is chosen
        else:
            h = self.h
            t = self.t0
            i = 0

            #Time stepping loop
            while t < self.tf:
                r_i = self.r_points[i]

                if self.adaptive:
                    #Full step
                    r1 = self.choosen_method(t, r_i, h)

                    #First half step
                    r_half = self.choosen_method(t, r_i, h / 2)

                    #Second half step
                    r2 = self.choosen_method(t + h / 2, r_half, h / 2)

                    #Estimate error and adjust step size
                    Err = np.max(np.abs(r2 - r1))
                    Err = max(Err, 1e-14) #Avoid divition by zero
                    h = h * (self.max_tol / Err)**self.step_exponent

                #Avoid stepping beyond tf
                h = min(h, self.tf - t)

                #Advance one step
                self.r = self.choosen_method(t, r_i, h)
                self.r_points[i + 1]= self.r.copy()
                t += h
                self.t_points[i + 1]= t
                i += 1

            #Trim unused preallocated arrays
            self.t_points = self.t_points[:i+1]
            self.r_points = self.r_points[:i+1]

        return self.t_points, self.r_points

