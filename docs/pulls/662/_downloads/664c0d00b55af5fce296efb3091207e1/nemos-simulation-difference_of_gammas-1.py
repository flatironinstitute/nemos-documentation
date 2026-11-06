import matplotlib.pyplot as plt
from nemos.simulation import difference_of_gammas
coupling_duration = 100
inhib_a, inhib_b = 1.0, 1.0
excit_a, excit_b = 2.0, 2.0
coupling_filter = difference_of_gammas(
    ws=coupling_duration,
    inhib_a=inhib_a,
    inhib_b=inhib_b,
    excit_a=excit_a,
    excit_b=excit_b
)
_ = plt.plot(coupling_filter)
_ = plt.title("Coupling filter from difference of gammas")
_ = plt.show()
