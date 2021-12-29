import numpy as np

from mean_variance_alg import *

np.set_printoptions(4, suppress=True)
part_switch = {
    "Part1": False,
    "Part2": True
}

if part_switch["Part1"]:
    # --- Part 1
    mu = np.array([0.8, 0.2, 0.5, -0.2])
    r = 0.2
    sigma = np.array([
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

    print("u0   = {:9.6f}".format(u0))
    print("u1   = {:9.6f}".format(u1))
    print("diff = {:9.6f}".format(diff))

# --- part 2
if part_switch["Part2"]:
    n = 1000
    x0 = np.random.normal(loc=0.1, scale=1, size=n)
    e = np.random.normal(loc=-0.05, scale=0.5, size=n)
    x1 = x0 + e
    x2 = -x0 - x1
    r = np.array([x0, x1, x2]).transpose()
    mu = r.mean(axis=0)
    sigma = np.cov(r, rowvar=False)
    lbd = 10
    _, p = r.shape
    w0 = np.ones(p) / p
    sec = np.array([[1, 1, 0], [0, 0, 1]])
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
