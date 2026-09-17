import numpy as np
import jax
import matplotlib.pyplot as plt
from nemos.simulation import simulate_recurrent
np.random.seed(42)
n_neurons = 2
coupling_duration = 100
feedforward_input = np.random.normal(size=(1000, n_neurons, 1))
coupling_basis = np.random.normal(size=(coupling_duration, 10))
coupling_coef = 0.5*np.random.normal(size=(n_neurons, n_neurons, 10))
intercept = -9 * np.ones(n_neurons)
init_spikes = np.zeros((coupling_duration, n_neurons))
random_key = jax.random.key(123)
spikes, rates = simulate_recurrent(
    coupling_coef=coupling_coef,
    feedforward_coef=np.ones((n_neurons, 1)),
    intercepts=intercept,
    random_key=random_key,
    feedforward_input=feedforward_input,
    coupling_basis_matrix=coupling_basis,
    init_y=init_spikes
)
_ = plt.figure()
_ = plt.plot(rates[:, 0], label="Neuron 0 rate")
_ = plt.plot(rates[:, 1], label="Neuron 1 rate")
_ = plt.legend()
_ = plt.title("Simulated firing rates")
_ = plt.show()
