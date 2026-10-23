"""
MSE de la regresión lineal con dos regresores y geometría del Lasso.
Panel izquierdo: superficie MSE(b1, b2).
Panel derecho: contornos del MSE + región de restricción |b1| + |b2| <= t.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

rng = np.random.default_rng(2026)

# ---------- Datos simulados ----------
n, rho = 200, 0.5
cov = np.array([[1, rho], [rho, 1]])
X = rng.multivariate_normal([0, 0], cov, size=n)
beta_true = np.array([1.5, 0.4])
y = X @ beta_true + rng.normal(0, 1, n)

# Estandarizar X y centrar y
X = (X - X.mean(0)) / X.std(0)
y = y - y.mean()

# ---------- MCO y Lasso ----------
b_ols = np.linalg.solve(X.T @ X, X.T @ y)
lam = 0.55
lasso = Lasso(alpha=lam, fit_intercept=False).fit(X, y)
b_las = lasso.coef_
t = np.abs(b_las).sum()             # presupuesto equivalente de la restricción

def mse(b1, b2):
    r = y[:, None, None] - X[:, [0]][:, :, None] * b1 - X[:, [1]][:, :, None] * b2
    return (r ** 2).mean(axis=0)

g = np.linspace(-1.0, 2.6, 300)
B1, B2 = np.meshgrid(g, g)
Z = (np.mean(y**2)
     - 2 * (B1 * np.mean(X[:, 0] * y) + B2 * np.mean(X[:, 1] * y))
     + B1**2 * np.mean(X[:, 0]**2) + B2**2 * np.mean(X[:, 1]**2)
     + 2 * B1 * B2 * np.mean(X[:, 0] * X[:, 1]))

mse_ols = np.mean((y - X @ b_ols) ** 2)
mse_las = np.mean((y - X @ b_las) ** 2)

# ---------- Figura ----------
fig = plt.figure(figsize=(14, 6.2))

# Panel 1: superficie
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(B1, B2, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
ax1.contour(B1, B2, Z, levels=15, cmap="viridis", offset=Z.min() - 0.5, linewidths=0.8)
ax1.scatter(*b_ols, mse_ols, color="red", s=50, depthshade=False)
ax1.text(b_ols[0], b_ols[1], mse_ols + 0.6, r"$\hat\beta^{MCO}$", color="red", fontsize=12)
ax1.set_xlabel(r"$\beta_1$", fontsize=12)
ax1.set_ylabel(r"$\beta_2$", fontsize=12)
ax1.set_zlabel("MSE", fontsize=11)
ax1.set_zlim(Z.min() - 0.5, Z.max())
ax1.set_title(r"Superficie del MSE: $\frac{1}{n}\sum (y_i-\beta_1x_{1i}-\beta_2x_{2i})^2$",
              fontsize=12)
ax1.view_init(elev=28, azim=-125)

# Panel 2: contornos + restricción L1
ax2 = fig.add_subplot(1, 2, 2)
levels = np.sort(np.concatenate([mse_ols + np.array([0.05, 0.3, 0.7, 1.2, 1.9, 2.8, 4]),
                                 [mse_las]]))
cs = ax2.contour(B1, B2, Z, levels=levels, cmap="viridis", linewidths=1.2)
ax2.contour(B1, B2, Z, levels=[mse_las], colors="darkorange", linewidths=2.5)

diamond = np.array([[t, 0], [0, t], [-t, 0], [0, -t], [t, 0]])
ax2.fill(diamond[:, 0], diamond[:, 1], color="steelblue", alpha=0.25)
ax2.plot(diamond[:, 0], diamond[:, 1], color="steelblue", lw=2,
         label=rf"Restricción $|\beta_1|+|\beta_2|\leq t$  ($t={t:.2f}$)")

ax2.plot(*b_ols, "o", color="red", ms=8,
         label=rf"$\hat\beta^{{MCO}}=({b_ols[0]:.2f},\ {b_ols[1]:.2f})$")
ax2.plot(*b_las, "s", color="darkorange", ms=9, mec="k",
         label=rf"$\hat\beta^{{Lasso}}=({b_las[0]:.2f},\ {b_las[1]:.2f})$")
ax2.plot([], [], color="darkorange", lw=2.5, label="Contorno tangente a la restricción")

ax2.axhline(0, color="gray", lw=0.7); ax2.axvline(0, color="gray", lw=0.7)
ax2.set_xlim(-1.0, 2.6); ax2.set_ylim(-1.0, 2.6)
ax2.set_aspect("equal")
ax2.set_xlabel(r"$\beta_1$", fontsize=12); ax2.set_ylabel(r"$\beta_2$", fontsize=12)
ax2.set_title(rf"Contornos del MSE y solución Lasso ($\lambda={lam}$)", fontsize=12)
ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

plt.tight_layout()
out = Path(__file__).resolve().parent / "images" / "lasso_mse_contornos.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print("MCO:", b_ols, "MSE", mse_ols)
print("Lasso:", b_las, "MSE", mse_las, "t", t)
