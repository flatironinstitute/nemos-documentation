import numpy as np
import matplotlib.pyplot as plt
from nemos.basis import BSplineConv
bspline_basis = BSplineConv(n_basis_funcs=4, order=3, window_size=10)
sample_points, basis_values = bspline_basis.evaluate_on_grid(100)
for i in range(4):
    p = plt.plot(sample_points, basis_values[:, i], label=f'Function {i+1}')
plt.title('B-Spline Basis Functions')
# Expected:
## Text(0.5, 1.0, 'B-Spline Basis Functions')
plt.xlabel('Domain')
# Expected:
## Text(0.5, 0, 'Domain')
plt.ylabel('Basis Function Value')
# Expected:
## Text(0, 0.5, 'Basis Function Value')
l = plt.legend()
plt.show()
