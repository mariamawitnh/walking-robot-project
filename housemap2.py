import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spatialmath as sm
from spatialgeometry import Cuboid
 
import roboticstoolbox as rt
from roboticstoolbox.backends.PyPlot import PyPlot
from scipy.io import loadmat
import networkx as nx
from scipy.spatial import KDTree
 
 
 
    """
    Probabilistic Road-Map planner.
    Bruker numpy/scipy/networkx – ingen roboticstoolbox.mobile nødvendig.
    """
 
    def __init__(self, occ_map, cell_size=0.01, npoints=5000,
                 k_neighbors=10, seed=42):
        """
        occ_map    : 2D numpy-array  (0 = fri, 1 = hindring)
        cell_size  : meter per celle (0.01 = 1 cm)
        npoints    : antall tilfeldige noder å sample
        k_neighbors: antall nærmeste naboer å forsøke koble til
        """
        self.occ_map   = occ_map
        self.cell_size = cell_size
        self.npoints   = npoints
        self.k         = k_neighbors
        self.rng       = np.random.default_rng(seed)
        self.graph     = nx.Graph()
 
        self.rows, self.cols = occ_map.shape
        self.width  = self.cols * cell_size   # meter
        self.height = self.rows * cell_size   # meter
 
    # ── Hjelpefunksjoner ─────────────────────────────────────────────────────
 
    def _xy_to_cell(self, x, y):
        """Konverter (x, y) i meter til (rad, kol) i kartet."""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return row, col
 
    def _is_free(self, x, y):
        """Sjekk om et punkt (x, y) i meter er i fri celle."""
        row, col = self._xy_to_cell(x, y)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.occ_map[row, col] == 0
 
    def _collision_free(self, p1, p2, n_checks=20):
        """
        Sjekk om linjen mellom p1 og p2 er hindringsfri.
        Deler linjen inn i n_checks punkter og sjekker hvert.
        """
        for t in np.linspace(0, 1, n_checks):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not self._is_free(x, y):
                return False
        return True
 
    # ── Bygg veikart ─────────────────────────────────────────────────────────
 
    def build(self):
        """Sample tilfeldige noder og koble dem til nabonoder uten kollisjoner."""
        print(f"Sampler {self.npoints} tilfeldige noder ... ", end="", flush=True)
 
        nodes = []
        attempts = 0
        while len(nodes) < self.npoints and attempts < self.npoints * 20:
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            if self._is_free(x, y):
                nodes.append((x, y))
            attempts += 1
 
        self.nodes = np.array(nodes)
        print(f"fikk {len(self.nodes)} noder.")
 
        for i, (x, y) in enumerate(self.nodes):
            self.graph.add_node(i, pos=(x, y))
 
        print(f"Kobler graf med k={self.k} naboer ... ", end="", flush=True)
        tree  = KDTree(self.nodes)
        edges = 0
        for i, node in enumerate(self.nodes):
            dists, idxs = tree.query(node, k=self.k + 1)
            for dist, j in zip(dists[1:], idxs[1:]):
                if not self.graph.has_edge(i, j):
                    if self._collision_free(self.nodes[i], self.nodes[j]):
                        self.graph.add_edge(i, j, weight=dist)
                        edges += 1
 
        print(f"ferdig  ({len(self.nodes)} noder, {edges} kanter).")
 
    # ── Sti-spørring ─────────────────────────────────────────────────────────
 
    def query(self, start_xy, goal_xy):
        """
        Finn korteste sti fra start_xy til goal_xy (begge i meter).
        Returnerer numpy-array med form (N, 2).
        """
        tree  = KDTree(self.nodes)
        s_idx = len(self.graph.nodes)
        g_idx = s_idx + 1
 
        self.graph.add_node(s_idx, pos=tuple(start_xy))
        self.graph.add_node(g_idx, pos=tuple(goal_xy))
 
        for temp_idx, pt in [(s_idx, start_xy), (g_idx, goal_xy)]:
            _, neighbors = tree.query(pt, k=self.k)
            for nb in neighbors:
                if self._collision_free(pt, self.nodes[nb]):
                    dist = np.linalg.norm(pt - self.nodes[nb])
                    self.graph.add_edge(temp_idx, nb, weight=dist)
 
        try:
            path_indices = nx.astar_path(
                self.graph, s_idx, g_idx,
                heuristic=lambda a, b: np.linalg.norm(
                    np.array(self.graph.nodes[a]["pos"]) -
                    np.array(self.graph.nodes[b]["pos"])
                ),
                weight="weight"
            )
            path_xy = np.array([self.graph.nodes[n]["pos"] for n in path_indices])
        except nx.NetworkXNoPath:
            raise RuntimeError("Ingen sti funnet – prøv høyere npoints eller k_neighbors")
        finally:
            self.graph.remove_node(s_idx)
            self.graph.remove_node(g_idx)
 
        return path_xy
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  STEG 2: Last kart og planlegg stier
# ─────────────────────────────────────────────────────────────────────────────
 
def load_house_map():
    """Laster house.mat og returnerer (occ_map, cell_size)."""
    try:
        data = rt.rtb_load_matfile("data/house.mat")
    except Exception:
        data = loadmat("house.mat")
 
    if "map" in data:
        raw = data["map"]
    elif "floorplan" in data:
        raw = data["floorplan"]
    else:
        raw = [v for v in data.values() if isinstance(v, np.ndarray)][0]
 
    raw = (raw > 0).astype(np.uint8)
 
    CELL_SIZE = 0.01
    print(f"Kart lastet:  {raw.shape[1]*CELL_SIZE:.2f} m x "
          f"{raw.shape[0]*CELL_SIZE:.2f} m  ({raw.shape} celler)")
    return raw, CELL_SIZE
 
 
def random_free_point(occ_map, cell_size, rng):
    """Returnerer tilfeldig (x, y) i meter i en fri celle."""
    rows, cols = np.where(occ_map == 0)
    idx  = rng.integers(len(rows))
    r, c = rows[idx], cols[idx]
    x = (c + 0.5) * cell_size
    y = (r + 0.5) * cell_size
    return np.array([x, y])
 
 
def plan_random_paths(prm, occ_map, cell_size, n_paths=5, seed=7):
    """Planlegger n_paths tilfeldige stier og returnerer liste av arrays."""
    rng   = np.random.default_rng(seed)
    paths = []
 
    print(f"\nPlanlegger {n_paths} tilfeldige stier:")
    for i in range(n_paths):
        start = random_free_point(occ_map, cell_size, rng)
        goal  = random_free_point(occ_map, cell_size, rng)
        try:
            path   = prm.query(start, goal)
            length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
            print(f"  Sti {i+1}: {np.round(start,2)} -> {np.round(goal,2)}"
                  f"  |  {length:.2f} m  |  {len(path)} veipunkter")
            paths.append(path)
        except RuntimeError as e:
            print(f"  Sti {i+1}: {e}")
 
    return paths
 
 
def plot_all_paths(prm, occ_map, cell_size, paths):
    """Tegner kartet, PRM-grafen og alle stier i én figur."""
    COLORS   = ["#E63946", "#2A9D8F", "#E9C46A", "#A8DADC", "#F4A261"]
    width_m  = occ_map.shape[1] * cell_size
    height_m = occ_map.shape[0] * cell_size
    extent   = [0, width_m, 0, height_m]
 
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(occ_map, cmap="gray_r", origin="lower",
              extent=extent, vmin=0, vmax=1, alpha=0.85)
 
    ax.scatter(prm.nodes[:, 0], prm.nodes[:, 1],
               s=1, c="steelblue", alpha=0.20, zorder=2, label="PRM-noder")
    for u, v in prm.graph.edges():
        if u < len(prm.nodes) and v < len(prm.nodes):
            n1, n2 = prm.nodes[u], prm.nodes[v]
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]],
                    color="steelblue", lw=0.25, alpha=0.12, zorder=2)
 
    patches = []
    for i, path in enumerate(paths):
        c = COLORS[i % len(COLORS)]
        ax.plot(path[:, 0], path[:, 1], color=c, lw=2.5, zorder=5,
                solid_capstyle="round")
        ax.plot(*path[0],  "o", color=c, ms=9,  zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(*path[-1], "*", color=c, ms=13, zorder=6,
                markeredgecolor="white", markeredgewidth=1.0)
        ax.annotate(f"S{i+1}", path[0],  color=c, fontsize=7, zorder=7,
                    xytext=(4, 4), textcoords="offset points", fontweight="bold")
        ax.annotate(f"G{i+1}", path[-1], color=c, fontsize=7, zorder=7,
                    xytext=(4, 4), textcoords="offset points", fontweight="bold")
        patches.append(mpatches.Patch(color=c, label=f"Sti {i+1}"))
 
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]")
    ax.set_title("PRM Path Planning – House map\n5 tilfeldige stier", fontsize=12)
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    plt.savefig("prm_paths_house.png", dpi=150, bbox_inches="tight")
    print("Rapportfigur lagret: prm_paths_house.png")
    plt.show()
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  STEG 3: WalkingRobot – følger PRM-veipunkter
# ─────────────────────────────────────────────────────────────────────────────
 
class WalkingRobot:
    def __init__(self, goal_list: list, dt_anim=0.02):
        print("\nInit robot...")
        mm = 0.001
        L1 = 100 * mm
        L2 = 100 * mm
 
        leg = rt.ERobot(
            rt.ET.Rz() * rt.ET.Rx() * rt.ET.ty(L1) * rt.ET.Rx() * rt.ET.tz(-L2)
        )
 
        xf, xb = 50, -50
        y = -50
        zu, zd = -20, -50
 
        segments = np.array([
            [xf, y, zd], [xb, y, zd], [xb, y, zu], [xf, y, zu], [xf, y, zd]
        ]) * mm
 
        x = rt.mstraj(segments, tsegment=[3, 0.25, 0.5, 0.25], dt=0.01, tacc=0.07)
 
        print("Invers kinematikk ... ", end="", flush=True)
        xcycle = x.q
        xcycle = np.vstack((xcycle, xcycle[-3:, :]))
        sol    = leg.ikine_LM(sm.SE3(xcycle), mask=[1, 1, 1, 0, 0, 0])
        print("ferdig")
        qcycle = sol.q
 
        cycle_time = 4.0
        body_vel   = (xf - xb) * mm / cycle_time
 
        total_path_len = sum(
            np.linalg.norm(np.array(goal_list[i+1]) - np.array(goal_list[i]))
            for i in range(len(goal_list) - 1)
        ) if len(goal_list) > 1 else 1.0
 
        n_steps = int(total_path_len * 1.5 / (body_vel * dt_anim)) + 2000
 
        W = 100 * mm
        L = 200 * mm
 
        leg_offsets = [
            sm.SE3( L/2, -W/2, 0),
            sm.SE3(-L/2, -W/2, 0),
            sm.SE3( L/2,  W/2, 0) * sm.SE3.Rz(np.pi),
            sm.SE3(-L/2,  W/2, 0) * sm.SE3.Rz(np.pi),
        ]
        legs = [rt.ERobot(leg, name=f"leg{i}") for i in range(4)]
 
        xs = [g[0] for g in goal_list]
        ys = [g[1] for g in goal_list]
        pad = L + 0.20
        env_lim_x = max(abs(max(xs)), abs(min(xs))) + pad
        env_lim_y = max(abs(max(ys)), abs(min(ys))) + pad
 
        env = PyPlot()
        env.launch(limits=[
            -env_lim_x, env_lim_x,
            -env_lim_y, env_lim_y,
            -0.15, 0.10
        ])
 
        start_x = goal_list[0][0]
        start_y = goal_list[0][1]
        T_wb    = sm.SE3(start_x, start_y, 0)
 
        for i, leg_robot in enumerate(legs):
            leg_robot.base = T_wb * leg_offsets[i]
            leg_robot.q    = np.r_[0, 0, 0]
            env.add(leg_robot, readonly=True, jointaxes=False,
                    eeframe=False, shadow=False)
 
        body = Cuboid([L, W, 30 * mm], color='b')
        body.base = T_wb
        env.add(body)
        env.step()
 
        pos_x, pos_y, theta = start_x, start_y, 0.0
        K_p        = 2.0
        goal_index = 1
 
        for i in range(n_steps):
            if not plt.fignum_exists(env.fig.number):
                break
            if goal_index >= len(goal_list):
                print(f"Alle mål nådd etter {i} steg!")
                break
 
            goal = goal_list[goal_index]
            dist = np.hypot(goal[0] - pos_x, goal[1] - pos_y)
 
            if dist < 0.025:
                print(f"  Veipunkt {goal_index}/{len(goal_list)-1} nådd")
                goal_index += 1
                continue
 
            bearing       = np.arctan2(goal[1] - pos_y, goal[0] - pos_x)
            heading_error = (bearing - theta + np.pi) % (2 * np.pi) - np.pi
            theta        += K_p * heading_error * dt_anim
            ds            = body_vel * dt_anim
            pos_x        += ds * np.cos(theta)
            pos_y        += ds * np.sin(theta)
 
            legs[0].q = self._gait(qcycle, i,   0, False)
            legs[1].q = self._gait(qcycle, i, 100, False)
            legs[2].q = self._gait(qcycle, i, 200, True)
            legs[3].q = self._gait(qcycle, i, 300, True)
 
            T_wb = sm.SE3(pos_x, pos_y, 0) * sm.SE3.Rz(theta)
            for j, leg_robot in enumerate(legs):
                leg_robot.base = T_wb * leg_offsets[j]
            body.base = T_wb
            env.step(dt=dt_anim)
 
        env.hold()
        plt.close('all')
 
    def _gait(self, cycle, k, offset, flip):
        k = (k + offset) % cycle.shape[0]
        q = cycle[k, :].copy()
        if flip:
            q[0] = -q[0]
        return q
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  KJØRING
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    # 1. Last kart
    occ_map, cell_size = load_house_map()
 
    # 2. Bygg PRM-veikart (gjøres kun én gang)
    prm = SimplePRM(occ_map, cell_size=cell_size, npoints=5000, k_neighbors=10)
    prm.build()
 
    # 3. Planlegg 5 tilfeldige stier
    paths = plan_random_paths(prm, occ_map, cell_size, n_paths=5)
 
    # 4. Tegn rapport-figur
    plot_all_paths(prm, occ_map, cell_size, paths)
 
    # 5. Kjør robot langs hver sti
    for path_nr, path in enumerate(paths):
        print(f"\n=== Robot følger sti {path_nr + 1} ===")
        goal_list = [(pt[0], pt[1]) for pt in path]
        WalkingRobot(goal_list=goal_list)