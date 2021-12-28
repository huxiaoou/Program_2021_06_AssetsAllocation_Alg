from mean_variance_alg import *

np.set_printoptions(4, suppress=True)

mu = np.array([0.8, 0.2, 0.5, -0.2])
r = 0.02
sigma = np.array([
    [1.0, r, r, r],
    [r, 1.0, r, r],
    [r, r, 1.0, r],
    [r, r, r, 1.0],
])
lbd = 0.001
w0 = np.ones(4) / 4
sec = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
sec_bound = sec.dot(w0)

print("\nminimize_utility_con6")
w, utility_min = minimize_utility_con6(
    t_mu=mu,
    t_sigma=sigma,
    t_lbd=lbd,
    t_bound=(0, 1),
    t_sec=sec,
    t_sec_bound=sec_bound,
)
if w is not None:
    print("w           = {}".format(w))
    print("utility_min = {:.6f}".format(utility_min))
else:
    print("ERROR")
