from mean_variance_alg import *

np.set_printoptions(4, suppress=True)

mu = np.array([0.8, 0.2, 0.5, -0.2])
r = 0.2
sigma = np.matrix([
    [1.0, r, r, r],
    [r, 1.0, r, r],
    [r, r, 1.0, r],
    [r, r, r, 1.0],
])
lbd = 1
w0 = np.ones(4) / 4
sec = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
sec_bound = sec.dot(w0)

print("\nminimize_utility_con6")
w, utility_min = minimize_utility_con6_cvxpy(
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

w1 = np.array([0.5, 0, 0.5, 0])
u0 = portfolio_utility(t_w=w, t_mu=mu, t_sigma=sigma, t_lbd=lbd)
u1 = portfolio_utility(t_w=w1, t_mu=mu, t_sigma=sigma, t_lbd=lbd)
diff = np.sum([
    abs(sec.dot(w0) - sec.dot(w1)).sum(),
]
)

print("u0   = {:9.6f}".format(u0[0, 0]))
print("u1   = {:9.6f}".format(u1[0, 0]))
print("diff = {:9.6f}".format(diff))
