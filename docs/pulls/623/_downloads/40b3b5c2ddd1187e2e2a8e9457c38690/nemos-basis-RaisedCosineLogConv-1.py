import numpy as np
import matplotlib.pyplot as plt
from nemos.basis import RaisedCosineLogConv
n_basis_funcs = 5
decay_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05]) # sample decay rates
window_size=10
ortho_basis = RaisedCosineLogConv(n_basis_funcs, window_size)
sample_points, basis_values = ortho_basis.evaluate_on_grid(100)
plt.plot(sample_points, basis_values)
# Expected:
## [<matplotlib.lines.Line2D object at ...
plt.show()
