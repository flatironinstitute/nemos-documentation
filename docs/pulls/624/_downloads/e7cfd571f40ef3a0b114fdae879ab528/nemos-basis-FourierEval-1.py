import numpy as np
import matplotlib.pyplot as plt
from nemos.basis import FourierEval
n_frequencies = 5
fourier_basis = FourierEval(n_frequencies)
sample_points, basis_values = fourier_basis.evaluate_on_grid(100)
plt.plot(sample_points, basis_values)
# Expected:
## [<matplotlib.lines.Line2D object at ...
plt.show()
