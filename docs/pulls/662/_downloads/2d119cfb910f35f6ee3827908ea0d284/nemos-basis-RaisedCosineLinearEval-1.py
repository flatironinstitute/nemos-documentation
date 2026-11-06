import numpy as np
import matplotlib.pyplot as plt
from nemos.basis import RaisedCosineLinearEval
n_basis_funcs = 5
window_size=10
raised_cos_basis = RaisedCosineLinearEval(n_basis_funcs)
sample_points, basis_values = raised_cos_basis.evaluate_on_grid(100)
plt.plot(sample_points, basis_values)
# Expected:
## [<matplotlib.lines.Line2D object at ...
plt.show()
