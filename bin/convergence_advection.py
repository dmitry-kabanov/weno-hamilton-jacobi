from wenohj.solver import Solver
import numpy as np
import matplotlib.pyplot as plt
import sys
from time import strftime


def get_alpha(x, t, u, u_x_plus, u_x_minus):
    """ Compute alpha for local Lax-Friedrichs flux. """
    max = np.ones_like(x)

    return max


def flux(x, t, u, u_x):
    return u_x


def ic(x):
    return np.sin(np.pi * x)


def exact_solution(x):
    return ic(x)

if len(sys.argv) <= 1 or sys.argv[1] not in ['compute', 'plot']:
    print('USAGE: convergence_advection_test.py <command>\n'
          'where <command>:\n'
          '  compute - to compute convergence rate and write results to file\n'
          '  plot    - to read results from file and plot figure\n')
    sys.exit(1)

filename = 'data/convergence_advection.txt'
lb = -1.0
rb = 1.0
T = 2.0

npoints = [40, 80, 160, 320, 640, 1280, 2560]
errors_L1 = []
errors_Linf = []

if sys.argv[1] == 'compute':
    for n in npoints:
        # Spatial step. We add 1 because n is the number of internal points.
        dx = (rb - lb) / (n + 1.0)
        cfl_number = dx
        s = Solver(lb, rb, n, flux, get_alpha, 'periodic', cfl=cfl_number)
        x = s.get_x()
        u0 = ic(x)
        solution = s.solve(u0, T)
        exact = exact_solution(x - T)
        error = dx * np.linalg.norm(exact - solution, 1)
        errors_L1.append(error)
        error = np.linalg.norm(exact - solution, np.inf)
        errors_Linf.append(error)

    with open(filename, mode='w', encoding='utf-8') as outfile:
        for i in range(len(errors_L1)):
            line = '{0:4d} {1:22.16e} {2:22.16e}'.format(
                npoints[i], errors_L1[i], errors_Linf[i])
            print(line)
            outfile.write(line + '\n')
else:
    with open(filename, mode='r', encoding='utf-8') as infile:
        for line in infile:
            _, error_L1, error_Linf = line.split()
            errors_L1.append(float(error_L1))
            errors_Linf.append(float(error_Linf))

        assert len(npoints) == len(errors_L1), \
            'Lengths of npoints and errors_L1 mismatch'
        assert len(npoints) == len(errors_Linf), \
            'Lengths of npoints and errors_Linf mismatch'

        print('{0:4d};{1:8.2e};{2:6s};{3:8.2e};{4:6s}'.format(
            npoints[0], errors_L1[0], '-', float(errors_Linf[0]), '-'))
        for i in range(1, len(npoints)):
            err_ratio = np.log(errors_L1[i] / errors_L1[i-1])
            step_ratio = np.log((npoints[i-1] + 1.0) / (npoints[i] + 1.0))
            p_L1 = err_ratio / step_ratio
            err_ratio = np.log(errors_Linf[i] / errors_Linf[i-1])
            step_ratio = np.log((npoints[i-1] + 1.0) / (npoints[i] + 1.0))
            p_Linf = err_ratio / step_ratio
            print('{0:4d};{1:8.2e};{2:4.2f};{3:8.2e};{4:4.2f}'.format(
                npoints[i], errors_L1[i], p_L1, errors_Linf[i], p_Linf))

    plt.loglog(npoints, errors_L1, '-o', label=r'$\|\| E \|\|_1$')
    n_list = [5e3 * n**(-5) for n in npoints]
    plt.loglog(npoints, n_list, '-s', label=r'$N^{-5}$')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$\|\| E \|\|_1$')
    plt.legend(loc='upper right')
    if len(sys.argv) == 3 and sys.argv[2] == 'save_fig':
        dt = strftime('%Y-%m-%dT%H%M%S')
        filename = 'images/convergence_advection_' + dt
        plt.savefig(filename)
    else:
        plt.show()
