import numpy as np
import matplotlib.pyplot as plt
from nemos.basis import IdentityEval
basis = IdentityEval()
sample_points, basis_values = basis.evaluate_on_grid(100)
plt.plot(sample_points, basis_values)
# Expected:
## [<matplotlib.lines.Line2D object at ...
plt.show()
