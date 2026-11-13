import jax
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
obs_low = nmo.observation_models.NegativeBinomialObservations(scale=0.05)
obs_high = nmo.observation_models.NegativeBinomialObservations(scale=2.0)
rate = np.full(1000, fill_value=10.0)
key = jax.random.PRNGKey(123)
samples_low = obs_low.sample_generator(key, rate)
samples_high = obs_high.sample_generator(key, rate)
bool(samples_high.var() > samples_low.var())
# Expected:
## True
_ = plt.subplot(211)
_, edges, _ = plt.hist(samples_high, bins=50)
_ = plt.title("scale = 2.0")
_ = plt.subplot(212)
_ = plt.hist(samples_low, bins=edges)
_ = plt.title("scale = 0.05")
plt.tight_layout()
plt.show()
