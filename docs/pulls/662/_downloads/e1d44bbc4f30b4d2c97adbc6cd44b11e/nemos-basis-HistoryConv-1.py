import matplotlib.pyplot as plt
from nemos.basis import HistoryConv
window_size=100
basis = HistoryConv(window_size=window_size)
sample_points, basis_values = basis.evaluate_on_grid(window_size)
plt.plot(sample_points, basis_values)
# Expected:
## [<matplotlib.lines.Line2D object at ...
plt.show()
