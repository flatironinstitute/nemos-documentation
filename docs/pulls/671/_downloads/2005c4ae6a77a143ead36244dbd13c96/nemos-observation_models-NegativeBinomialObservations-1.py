import jax
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
obs = nmo.observation_models.NegativeBinomialObservations()
rate = np.full(1000, fill_value=10.0)
key = jax.random.PRNGKey(123)
samples_low = obs.sample_generator(key, rate, dispersion=0.05)
samples_high = obs.sample_generator(key, rate, dispersion=2.0)
bool(samples_high.var() > samples_low.var())
# Expected:
## True
_ = plt.subplot(211)
_, edges, _ = plt.hist(samples_high, bins=50)
_ = plt.title("dispersion = 2.0")
_ = plt.subplot(212)
_ = plt.hist(samples_low, bins=edges)
_ = plt.title("dispersion = 0.05")
plt.tight_layout()
plt.show()
