import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat
 
# ── roboticstoolbox ──────────────────────────────────────────────────────────
import roboticstoolbox as rtb
from roboticstoolbox.mobile import PRM         # Probabilistic Road-Map
from roboticstoolbox.mobile import OccupancyGrid
 
# ── 1. Last inn kartet ───────────────────────────────────────────────────────
# Kartet er inkludert i Toolbox-pakken
data = rtb.rtb_load_matfile("data/house.mat")
 
# Kartet heter vanligvis 'map' eller 'floorplan' inne i .mat-filen
# Prøv begge varianter:
if "map" in data:
    raw_map = data["map"]
elif "floorplan" in data:
    raw_map = data["floorplan"]
else:
    # Ta første array-verdi i filen
    raw_map = list(data.values())[0]
 
# OccupancyGrid: cellestørrelse = 0.01 m (1 cm), opprinnelse (0, 0)
CELL_SIZE = 0.01   # meter per celle
og = OccupancyGrid(raw_map.astype(float), cellsize=CELL_SIZE)
 
print(f"Kartstørrelse:  {raw_map.shape[1] * CELL_SIZE:.2f} m (bredde)  x "
      f"{raw_map.shape[0] * CELL_SIZE:.2f} m (høyde)")
print(f"Antall celler:  {raw_map.shape}")
 
# ── 2. Bygg PRM-veikart ──────────────────────────────────────────────────────
# npoints: antall tilfeldige noder
# distthresh: maks avstand mellom noder for å legge til kant (meter)
# Øk npoints til kartet er godt tilkoblet (alle rom nås)
N_POINTS    = 5000   # juster opp om grafen ikke dekker alle rom
DIST_THRESH = 0.40   # meter – tilsvarer 40 cm maksimal kantlengde
 
prm = PRM(og, npoints=N_POINTS, distthresh=DIST_THRESH, seed=42)
prm.plan(showsamples=False)
 
print(f"\nPRM-graf:  {prm.graph.n()} noder,  {prm.graph.ne()} kanter")
 
# ── 3. Hjelp-funksjon: tilfeldig fri celle ───────────────────────────────────
rng = np.random.default_rng(seed=7)
 
def random_free_point(og: OccupancyGrid) -> np.ndarray:
    """Returnerer et tilfeldig punkt (x, y) i meter som ligger i fri celle."""
    rows, cols = np.where(og.grid == 0)          # 0 = fri, 1 = hindring
    while True:
        idx = rng.integers(len(rows))
        r, c = rows[idx], cols[idx]
        # Konverter rad/kolonne til meter (senter av celle)
        x = (c + 0.5) * og.cellsize + og.origin[0]
        y = (r + 0.5) * og.cellsize + og.origin[1]
        return np.array([x, y])
 
# ── 4. Plan 5 tilfeldige stier med samme veikart ────────────────────────────
N_PATHS = 5
paths   = []
points  = []          # lista med (start, goal) par
 
print("\n--- Planlegger 5 tilfeldige stier ---")
for i in range(N_PATHS):
    start = random_free_point(og)
    goal  = random_free_point(og)
    try:
        path = prm.query(start, goal)
        paths.append(path)
        points.append((start, goal))
        length_m = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        print(f"  Sti {i+1}: start={np.round(start,2)}, mål={np.round(goal,2)}, "
              f"lengde={length_m:.2f} m, punkter={len(path)}")
    except Exception as e:
        print(f"  Sti {i+1}: FEIL – {e}  (prøv høyere npoints)")
        paths.append(None)
        points.append((start, goal))
 
# ── 5. Visualisering: ett samlet figur med alle 5 stier ─────────────────────
COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#A8DADC", "#F4A261"]
 
fig, ax = plt.subplots(figsize=(10, 14))
 
# -- Kartet (vend Y-aksen slik at rad 0 er nederst) --------------------------
extent = [
    og.origin[0],
    og.origin[0] + og.grid.shape[1] * og.cellsize,
    og.origin[1],
    og.origin[1] + og.grid.shape[0] * og.cellsize,
]
ax.imshow(
    og.grid,
    cmap="gray_r",          # svart = hindring, hvit = fri
    origin="lower",
    extent=extent,
    vmin=0, vmax=1,
    alpha=0.85,
)
 
# -- PRM-noder og kanter (dempet bakgrunn) ------------------------------------
# Noder
nodes_xy = np.array([[v.coord[0], v.coord[1]] for v in prm.graph])
ax.scatter(nodes_xy[:, 0], nodes_xy[:, 1],
           s=1, c="steelblue", alpha=0.25, zorder=2, label="PRM-noder")
 
# Kanter
for e in prm.graph.edges():
    u, v = e
    ax.plot([u.coord[0], v.coord[0]],
            [u.coord[1], v.coord[1]],
            color="steelblue", lw=0.3, alpha=0.15, zorder=2)
 
# -- Stier, start- og målpunkter ----------------------------------------------
legend_patches = []
for i, (path, (start, goal)) in enumerate(zip(paths, points)):
    c = COLORS[i]
    label = f"Sti {i+1}"
    if path is not None:
        ax.plot(path[:, 0], path[:, 1],
                color=c, lw=2.2, zorder=5, solid_capstyle="round")
 
    # Start: fylt sirkel
    ax.plot(*start, "o", color=c, ms=9, zorder=6,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(f"S{i+1}", start, fontsize=7, color=c, zorder=7,
                xytext=(4, 4), textcoords="offset points", fontweight="bold")
 
    # Mål: fylt stjerne
    ax.plot(*goal, "*", color=c, ms=13, zorder=6,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.annotate(f"G{i+1}", goal, fontsize=7, color=c, zorder=7,
                xytext=(4, 4), textcoords="offset points", fontweight="bold")
 
    legend_patches.append(mpatches.Patch(color=c, label=label))
 
# -- Fast skalering: hele huset vises ----------------------------------------
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_aspect("equal")
ax.set_xlabel("x  [m]", fontsize=11)
ax.set_ylabel("y  [m]", fontsize=11)
ax.set_title(
    f"PRM Path Planning – House map  "
    f"({N_POINTS} noder, distthresh={DIST_THRESH} m)\n"
    f"5 tilfeldige stier med samme veikart  |  celle = {CELL_SIZE*100:.0f} cm",
    fontsize=12,
)
 
# Forklaring
node_patch  = mpatches.Patch(color="steelblue", alpha=0.5, label="PRM-noder/kanter")
start_patch = plt.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor="gray", markersize=8, label="Start (S)")
goal_patch  = plt.Line2D([0], [0], marker="*", color="w",
                         markerfacecolor="gray", markersize=11, label="Mål (G)")
ax.legend(handles=legend_patches + [node_patch, start_patch, goal_patch],
          loc="upper right", fontsize=8, framealpha=0.85)
 
plt.tight_layout()
plt.savefig("prm_paths_house.png", dpi=150, bbox_inches="tight")
print("\nFigur lagret: prm_paths_house.png")
plt.show()
 
# ── 6. Tabell: oppsummering av stier ─────────────────────────────────────────
print("\n=== Oppsummering ===")
print(f"{'Sti':<5} {'Start (x,y) m':<22} {'Mål (x,y) m':<22} "
      f"{'Lengde [m]':<12} {'Punkter'}")
print("-" * 75)
for i, (path, (start, goal)) in enumerate(zip(paths, points)):
    if path is not None:
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        print(f"{i+1:<5} ({start[0]:.2f}, {start[1]:.2f}){'':<10}"
              f"({goal[0]:.2f}, {goal[1]:.2f}){'':<10}"
              f"{length:<12.2f} {len(path)}")
    else:
        print(f"{i+1:<5} {'INGEN STI FUNNET':^60}")