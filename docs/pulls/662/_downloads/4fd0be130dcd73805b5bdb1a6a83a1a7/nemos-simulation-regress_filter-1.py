import numpy as np
import matplotlib.pyplot as plt
from nemos.simulation import regress_filter, difference_of_gammas
from nemos.basis import RaisedCosineLogEval
filter_duration = 100
n_basis_funcs = 20
filter_bank = difference_of_gammas(filter_duration).reshape(filter_duration, 1, 1)
_, basis = RaisedCosineLogEval(10).evaluate_on_grid(filter_duration)
weights = regress_filter(filter_bank, basis)[0, 0]
print("Weights shape:", weights.shape)
# Expected:
## Weights shape: (10,)
_ = plt.plot(filter_bank[:, 0, 0], label=f"True filter")
_ = plt.plot(basis.dot(weights), "--", label=f"Approx. filter")
_ = plt.legend()
_ = plt.title("True vs. Approximated Filters")
_ = plt.show()
