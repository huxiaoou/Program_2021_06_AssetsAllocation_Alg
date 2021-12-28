from mean_variance_alg import *

np.set_printoptions(4, suppress=True)

mu = np.array([0.8, -0.2, 0.3])
sigma = np.array([
    [1, 0.5, -0.3],
    [0.5, 2, -0.1],
    [-0.3, -0.1, 3],
])
lbd = 1
confined_variance = 1
confined_return = 0.5
H = np.array([
    np.ones(len(mu))
])
h = np.array([
    1
])
F = np.concatenate([
    np.diag(np.ones(len(mu))),
    -np.diag(np.ones(len(mu))),
], axis=0)
f = np.concatenate([
    np.ones(len(mu)),
    np.zeros(len(mu)),
], axis=0)

# # -------------------------------------------------------------
# print("\nminimize_variance")
# w, var_min = minimize_variance(t_sigma=sigma)
# print("w       = {}".format(w))
# print("var_min = {:.6f}".format(var_min))
#
# print("\nminimize_variance_con")
# w, var_min = minimize_variance_con(t_sigma=sigma)
# print("w       = {}".format(w))
# print("var_min = {:.6f}".format(var_min))

# -------------------------------------------------------------
print("\nminimize_utility")
w, utility_min = minimize_utility(t_mu=mu, t_sigma=sigma, t_lbd=lbd)
print("w           = {}".format(w))
print("utility_min = {:.6f}".format(utility_min))

print("\nminimize_utility_con")
w, utility_min = minimize_utility_con(t_mu=mu, t_sigma=sigma, t_lbd=lbd)
print("w           = {}".format(w))
print("utility_min = {:.6f}".format(utility_min))

print("\nminimize_utility_con_analytic")
w, utility_min = minimize_utility_con_analytic(t_mu=mu, t_sigma=sigma, t_lbd=lbd, t_H=H, t_h=h, t_F=F, t_f=f)
print("w           = {}".format(w))
print("utility_min = {:.6f}".format(utility_min))

# print("\nminimize_utility_con2")
# w, utility_min = minimize_utility_con2(t_mu=mu, t_sigma=sigma, t_lbd=lbd)
# print("w           = {}".format(w))
# print("utility_min = {:.6f}".format(utility_min))

# # -------------------------------------------------------------
# print("\nminimize_risk_budget_con")
# rb = np.ones(shape=mu.shape)
# w, trc_min = minimize_risk_budget_con(t_sigma=sigma, t_rb=rb, t_verbose=False)
# print("w       = {}".format(w))
# print("TRC     = {:.12f}".format(trc_min))
# portfolio_risk_budget(t_w=w, t_sigma=sigma, t_rb=rb, t_verbose=True)
#
# print("\nminimize_risk_budget_con2")
# rb = np.ones(shape=mu.shape)
# w, trc_min = minimize_risk_budget_con2(t_sigma=sigma, t_rb=rb, t_verbose=False)
# print("w       = {}".format(w))
# print("TRC     = {:.12f}".format(trc_min))
# portfolio_risk_budget(t_w=w, t_sigma=sigma, t_rb=rb, t_verbose=True)
#
# print("\nminimize_risk_budget_con")
# rb = np.array([10, 1, 2])
# w, trc_min = minimize_risk_budget_con(t_sigma=sigma, t_rb=rb, t_verbose=False)
# print("w       = {}".format(w))
# print("TRC     = {:.12f}".format(trc_min))
# portfolio_risk_budget(t_w=w, t_sigma=sigma, t_rb=rb, t_verbose=True)
#
# print("\nminimize_risk_budget_con2")
# rb = np.array([10, 1, 2])
# w, trc_min = minimize_risk_budget_con2(t_sigma=sigma, t_rb=rb, t_verbose=False)
# print("w       = {}".format(w))
# print("TRC     = {:.12f}".format(trc_min))
# portfolio_risk_budget(t_w=w, t_sigma=sigma, t_rb=rb, t_verbose=True)
#
# # -------------------------------------------------------------
# print("\nmaximum return with confined variance")
# w, maximum_return = maximum_return_with_confined_variance(t_mu=mu, t_sigma=sigma, t_benchmark_s2=confined_variance)
# if w is not None:
#     print("w       = {}".format(w))
#     print("confined variance = {:.6f}".format(confined_variance))
#     print("maximum  return   = {:.6f}".format(-maximum_return))
#
# print("\nminimum variance with confined return")
# w, minimum_variance = minimum_variance_with_confined_return(t_mu=mu, t_sigma=sigma, t_benchmark_mu=confined_return)
# if w is not None:
#     print("w       = {}".format(w))
#     print("minimum variance  = {:.6f}".format(minimum_variance))
#     print("confined return   = {:.6f}".format(confined_return))
