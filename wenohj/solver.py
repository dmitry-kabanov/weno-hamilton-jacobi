import numpy as np


class Solver(object):
    def __init__(self, lb, rb, npoints, flux, get_alpha, bt, cfl=0.6):
        """@todo: Docstring for __init__.

        :ncells: @todo
        :returns: @todo

        """
        self.lb = lb
        self.rb = rb
        self.npoints = npoints
        self.flux = flux
        self.get_alpha = get_alpha
        self.boundary_type = 'periodic'

        self.nb = 3
        self.eps = 1.0e-6
        self.cfl = cfl

        self.dx = (rb - lb) / (npoints + 1.0 + 0.0)
        self.x = np.linspace(
            lb - self.nb * self.dx,
            rb + self.nb * self.dx,
            npoints + 2 + 2*self.nb)

        # Index of the leftmost point of the physical domain.
        self.left = self.nb
        # Index of the rightmost point of the physical domain.
        self.right = self.nb + self.npoints + 1

        self.c1 = 1.0 / 3.0
        self.c2 = 1.0 / 6.0

    def solve(self, ic, final_time):
        u0 = np.zeros_like(self.x)
        u1 = np.zeros_like(self.x)
        u2 = np.zeros_like(self.x)
        l1 = self.left
        l2 = self.right + 1
        u0[l1:l2] = ic
        self.t = 0
        self.dt = self.cfl * self.dx

        while (self.t < final_time):
            if (self.t + self.dt > final_time):
                self.dt = final_time - self.t

            self.t += self.dt

            u1 = u0 + self.dt * self._rhs(u0)
            u2 = (3.0 * u0 + u1 + self.dt * self._rhs(u1)) / 4.0
            u0 = (u0 + 2.0 * u2 + 2.0 * self.dt * self._rhs(u2)) / 3.0

        return u0[self.left:self.right+1]

    def get_x(self):
        return self.x[self.left:self.right + 1]

    def _rhs(self, u):
        self.apply_boundary_conditions(u)
        rhs_values = np.zeros_like(u)
        l1 = self.left
        l2 = self.right + 1

        der1 = u[l1-1:l2-1] - u[l1-2:l2-2]
        der2 = u[l1:l2] - u[l1-1:l2-1]
        der3 = u[l1+1:l2+1] - u[l1:l2]
        der4 = u[l1+2:l2+2] - u[l1+1:l2+1]
        numer = -der1 + 7 * der2 + 7 * der3 - der4
        common = numer / (12.0 * self.dx)

        # Compute second derivatives
        der1 = (u[l1+3:l2+3] - 2*u[l1+2:l2+2] + u[l1+1:l2+1]) / self.dx
        der2 = (u[l1+2:l2+2] - 2*u[l1+1:l2+1] + u[l1:l2]) / self.dx
        der3 = (u[l1+1:l2+1] - 2*u[l1:l2] + u[l1-1:l2-1]) / self.dx
        der4 = (u[l1:l2] - 2*u[l1-1:l2-1] + u[l1-2:l2-2]) / self.dx
        der5 = (u[l1-1:l2-1] - 2*u[l1-2:l2-2] + u[l1-3:l2-3]) / self.dx

        weno_plus_flux = self.weno_flux(der1, der2, der3, der4)
        u_x_plus = common + weno_plus_flux
        weno_minus_flux = self.weno_flux(der5, der4, der3, der2)
        u_x_minus = common - weno_minus_flux

        rhs_values[l1:l2] = -self.numerical_flux(u, u_x_plus, u_x_minus)
        return rhs_values

    def weno_flux(self, a, b, c, d):
        """Calculate WENO approximation of the flux.
        :returns: @todo

        """
        is0 = 13*(a - b)**2 + 3*(a - 3*b)**2
        is1 = 13*(b - c)**2 + 3*(b + c)**2
        is2 = 13*(c - d)**2 + 3*(3*c - d)**2

        alpha0 = 1.0 / (self.eps + is0)**2
        alpha1 = 6.0 / (self.eps + is1)**2
        alpha2 = 3.0 / (self.eps + is2)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        w0 = alpha0 / sum_alpha
        w2 = alpha2 / sum_alpha

        result = self.c1*w0*(a - 2*b + c) + self.c2*(w2 - 0.5)*(b - 2*c + d)
        return result

    def numerical_flux(self, u, u_x_plus, u_x_minus):
        """Compute numerical flux using Lax-Friedrichs approximation.

        :u: @todo
        :u_x_plus: @todo
        :u_x_minus: @todo
        :returns: @todo

        """
        l1 = self.left
        l2 = self.right + 1
        alpha = self.get_alpha(self.x[l1:l2], self.t, u[l1:l2],
                               u_x_plus, u_x_minus)
        avg = (u_x_plus + u_x_minus) / 2.0
        term1 = self.flux(
            self.x[l1:l2], self.t,
            u[l1:l2],
            avg)
        term2 = -alpha * (u_x_plus - u_x_minus) / 2.0
        return term1 + term2

    def apply_boundary_conditions(self, u):
        if self.boundary_type == 'periodic':
            for i in range(0, self.nb + 1):
                u[self.left - i] = u[self.right - i]
                u[self.right + i] = u[self.left + i]

            lp = self.nb
            rp = self.nb + self.npoints + 1
            assert u[lp] == u[rp]
            assert u[lp - 1] == u[rp - 1]
            assert u[lp - 2] == u[rp - 2]
            assert u[lp - 3] == u[rp - 3]
            assert u[rp + 1] == u[lp + 1]
            assert u[rp + 2] == u[lp + 2]
            assert u[rp + 3] == u[lp + 3]
