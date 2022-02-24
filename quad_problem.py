# Import packages.
import cvxpy as cp
import numpy as np
import pandas as pd

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 0)


def gen_H(t_m: int, t_n: int):
    _H = np.zeros(shape=(t_m, t_n))
    _id = np.random.randint(0, t_m, t_n)
    for i in range(t_n):
        _H[_id[i], i] = 1
    return _H


# Generate a random non-trivial quadratic program.
n = 30  # number of variables
m = 5  # number of sectors
np.random.seed(1)

s = np.random.randn(n, n)
sigma = s.T @ s
mu = np.random.randn(n)
lbd = 0.01

wb = np.ones(n) / n

H = gen_H(t_m=m, t_n=n)  # (m, n)
h = H @ wb  # (m, 1)
d = 0.05
h0, h1 = h - d, h + d

G = np.ones(shape=(1, n))  # (p, n)
g = np.ones(shape=(1, 1))  # (p,1)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob0 = cp.Problem(
    cp.Minimize(- 2 / lbd * mu @ x + cp.quad_form(x, sigma)),
    [H @ x <= h1, H @ x >= h0, G @ x == g, x >= 0]
)
prob0.solve()
wp0 = x.value

prob1 = cp.Problem(
    cp.Minimize(- 2 / lbd * mu @ x + cp.quad_form(x, sigma)),
    [H @ x <= h1, H @ x >= h0, cp.norm(x, 1) <= 1]
)
prob1.solve()
wp1 = x.value

# question
prob2 = cp.Problem(
    cp.Minimize(- 2 / lbd * mu @ x + cp.quad_form(x, sigma)),
    [H @ x == h, cp.norm(x, 1) <= 1]
)
prob2.solve()
wp2 = x.value

print(pd.DataFrame({
    "h-": h0,
    "h": h,
    "hp0": H @ wp0,
    "hp1": H @ wp1,
    "hp2": H @ wp2,
    "h+": h1,
}))

print(pd.DataFrame({
    "mu": mu,
    "wp0": wp0,
    "wp1": wp0,
    "wp2": wp0,
}))

# print(pd.DataFrame(
#     data=sigma,
#     index=["X{:02d}".format(z) for z in range(n)],
#     columns=["X{:02d}".format(z) for z in range(n)]
# ))

print("\n")
print("     x0.sum() = {:.6f}".format(wp0.sum()))
print("     x1.sum() = {:.6f}".format(wp1.sum()))
print("     x2.sum() = {:.6f}".format(wp2.sum()))
print("abs(x0).sum() = {:.6f}".format(np.abs(wp0).sum()))
print("abs(x1).sum() = {:.6f}".format(np.abs(wp1).sum()))
print("abs(x2).sum() = {:.6f}".format(np.abs(wp2).sum()))

print("The optimal value of Problem0 is {:.4f}", prob0.value)
print("The optimal value of Problem1 is {:.4f}", prob1.value)
print("The optimal value of Problem2 is {:.4f}", prob2.value)
