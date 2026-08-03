import jax, jax.numpy as jnp, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dtmm import (reflectance, rgb_pixel, crb_grid, GRAPHENE_LAYER_NM)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

# ---- Panel 1: validation — monolayer contrast vs oxide thickness ----
tox = jnp.linspace(20, 400, 381)
r0 = jax.vmap(lambda t: reflectance(0.0, t, 550.0))(tox)
r1 = jax.vmap(lambda t: reflectance(GRAPHENE_LAYER_NM, t, 550.0))(tox)
C = np.array((r0 - r1) / r0)
ax = axes[0]
ax.plot(np.array(tox), C, lw=2, color="#2166ac")
for x in (90, 285):
    ax.axvline(x, color="#b2182b", ls="--", lw=1, alpha=0.7)
ax.text(90, ax.get_ylim()[1]*0.02 + max(C)*0.92, " 90 nm", color="#b2182b", fontsize=9)
ax.text(285, max(C)*0.92, " 285 nm", color="#b2182b", fontsize=9)
ax.set_xlabel("SiO$_2$ thickness (nm)")
ax.set_ylabel("Monolayer graphene contrast @ 550 nm")
ax.set_title("Validation: contrast peaks vs. literature\n(dashed = standard wafer choices)")
ax.grid(alpha=0.25)

# ---- Panel 2: CRB identifiability heatmap ----
d_grid = jnp.linspace(GRAPHENE_LAYER_NM, 120.0, 240)
t_grid = jnp.linspace(50.0, 350.0, 240)
crb = np.array(crb_grid(d_grid, t_grid))  # (t, d), nm
ax = axes[1]
im = ax.pcolormesh(np.array(d_grid), np.array(t_grid), np.clip(crb, 1e-3, 1e2),
                   norm=LogNorm(vmin=1e-2, vmax=1e2), cmap="viridis_r", shading="auto")
cs = ax.contour(np.array(d_grid), np.array(t_grid), crb, levels=[1.0, 5.8],
                colors=["white", "#ff7f00"], linewidths=[1.2, 2.0])
ax.clabel(cs, fmt={1.0: "1 nm", 5.8: "5.8 nm ($\\varphi$-Adapt)"}, fontsize=8)
plt.colorbar(im, ax=ax, label="Cramér–Rao bound on thickness (nm)")
ax.set_xlabel("Graphene thickness (nm)")
ax.set_ylabel("SiO$_2$ thickness (nm)")
ax.set_title("Identifiability map: CRB($d$, $t_{ox}$)\nRGB, 0.5% noise — orange = current SOTA error")

# ---- Panel 3: metamerism / non-injectivity structure ----
d_line = jnp.linspace(GRAPHENE_LAYER_NM, 120.0, 300)
rgbs = np.array(jax.vmap(lambda d: rgb_pixel(d, 285.0))(d_line))  # (N,3)
D = np.linalg.norm(rgbs[:, None, :] - rgbs[None, :, :], axis=-1)
ax = axes[2]
im = ax.pcolormesh(np.array(d_line), np.array(d_line), D, cmap="magma", shading="auto")
plt.colorbar(im, ax=ax, label="$\\|RGB(d) - RGB(d')\\|_2$")
ax.set_xlabel("$d$ (nm)")
ax.set_ylabel("$d'$ (nm)")
ax.set_title("Metamerism structure @ $t_{ox}$=285 nm\ndark off-diagonal = ambiguous thickness pairs")

plt.tight_layout()
plt.savefig("feasibility_study.png", dpi=160)
print("saved. CRB stats: median=%.3f nm, %% of grid below 5.8nm SOTA = %.1f%%, below 1nm = %.1f%%"
      % (np.median(crb), 100*np.mean(crb < 5.8), 100*np.mean(crb < 1.0)))
