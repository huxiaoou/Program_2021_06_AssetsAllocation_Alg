import sys
from mean_variance_alg import *

np.set_printoptions(4, suppress=True)
part_switch = {
    "Part1": True,
    "Part2": True
}

if part_switch["Part1"]:
    # --- Part 1
    mu = np.array([0.8, 0.2, 0.5, -0.2])
    r = 0
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
    w_opt, utility_opt = minimize_utility_con7_cvxpy(
        t_mu=mu,
        t_sigma=sigma,
        t_lbd=lbd,
        t_bound=(-1, 1),
        t_sec=sec,
        t_sec_bound=sec_bound,
        t_l_bound_offset=0,
        t_r_bound_offset=0
    )
    if w_opt is not None:
        print("w           = {}".format(w_opt))
        print("utility_min = {:.6f}".format(utility_opt))
        w_man = np.array([0.5, 0, 0.5, 0])
        u_opt = portfolio_utility(t_w=w_opt, t_mu=mu, t_sigma=sigma, t_lbd=lbd)
        u_man = portfolio_utility(t_w=w_man, t_mu=mu, t_sigma=sigma, t_lbd=lbd)
        diff = np.sum([
            abs(sec.dot(w0) - sec.dot(w_man)).sum(),
        ]
        )

        print("u_opt   = {:9.6f}".format(u_opt))
        print("u_man   = {:9.6f}".format(u_man))
        print("diff = {:9.6f}".format(diff))
    else:
        print("ERROR")

# --- part 2
if part_switch["Part2"]:
    n = 1000
    x0 = np.random.normal(loc=0.1, scale=1, size=n)
    x1 = -x0
    e2 = np.random.normal(loc=-0.1, scale=0.5, size=n)
    e3 = np.random.normal(loc=0.2, scale=0.5, size=n)
    x2 = x0 + e2
    x3 = x0 + e3
    x4 = -x0 - x1 - x2 - x3
    r = np.array([x0, x1, x2, x3, x4]).transpose()
    mu = r.mean(axis=0)
    sigma = np.cov(r, rowvar=False)
    lbd = float(sys.argv[1])
    _, p = r.shape
    w0 = np.ones(p) / p
    sec = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]])
    sec_bound = sec.dot(w0)

    print("\nminimize_utility_con6")
    w_opt, utility_opt = minimize_utility_con6_cvxpy(
        t_mu=mu,
        t_sigma=sigma,
        t_lbd=lbd,
        t_bound=(0, 1),
        t_sec=sec,
        t_sec_bound=sec_bound,
    )
    utility_raw = portfolio_utility(t_w=w0, t_mu=mu, t_sigma=sigma, t_lbd=lbd)
    if w_opt is not None:
        print("w           = {}".format(w_opt))
        print("raw utility = {:.6f}".format(utility_raw))
        print("opt utility = {:.6f}".format(utility_opt))
    else:
        print("ERROR")
