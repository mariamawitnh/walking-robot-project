""" Serves as the file for abstracting setting everything up """
import roboticstoolbox as rt
import numpy as np
import matplotlib.pyplot as plt
import spatialmath as sm
from spatialgeometry import Cuboid
from roboticstoolbox.backends.PyPlot import PyPlot


class WalkingRobot:
    """ A class for reusing the robot things, units in metres (usually) """
    GOAL_RADIUS = 0.05   # 5cm — waypoints are ~10cm apart in world coords
    BODY_WIDTH = 0.100
    BODY_LENGTH = 0.200
    L1 = L2 = 0.200

    def __init__(self,
                 goal_list: list = None,
                 path=None,
                 dt_anim=0.01,
                 cycle_time=4.01,
                 anim_skip_every=3,
                 floor_plan=None,
                 scale=0.01,
                 topdown=False,
                 cam_dist=0.4,
                 follow_cam=True
                 ):
        """ For some reason, ``cycle_time`` needs to be a float. It can never be an int """
        self._scale = scale

        if path is None:
            self._goal_list = [(p[0] * scale, p[1] * scale) for p in goal_list]
            self.path = None
        else:
            # path is Nx2 where columns are [row, col] (image coords)
            # convert to world coords: x = col * scale, y = row * scale
            self._goal_list = [(p[0] * scale, p[1] * scale) for p in path]
            self.path = path

        self._dt = dt_anim
        self._gait_dt = 0.01
        self._step_size = max(1, round(dt_anim / self._gait_dt))
        self._cam_dist = cam_dist

        self.leg = self._create_leg()
        self.legs = self._create_legs(self.leg)
        self.leg_offsets = self._create_leg_offsets()
        self.topdown = topdown

        self._skip = anim_skip_every
        self._follow_cam = follow_cam

        print("Init...")
        mm = 0.001

        xf = 50
        xb = -xf
        y = -50
        zu = -20
        zd = -50

        segments = np.array([
            [xf, y, zd],
            [xb, y, zd],
            [xb, y, zu],
            [xf, y, zu],
            [xf, y, zd]
        ]) * mm

        print("create trajectory\n")
        x = rt.mstraj(segments, tsegment=[3, 0.25, 0.5, 0.25], dt=self._gait_dt, tacc=0.07)

        print("inverse kinematics (this will take a moment)....", end='')
        xcycle = x.q
        xcycle = np.vstack((xcycle, xcycle[-3:, :]))
        sol = self.leg.ikine_LM(sm.SE3(xcycle), mask=[1, 1, 1, 0, 0, 0])
        print("done")
        self.qcycle = sol.q

        stroke_len = (xf - xb) * mm
        self.body_vel = stroke_len / cycle_time

        L = 200 * mm

        # Determine environment limits from floorplan or fallback
        if floor_plan is not None:
            h, w = floor_plan.shape[:2]
            world_w = w * scale
            world_h = h * scale
            x_min, x_max = 0, world_w
            y_min, y_max = 0, world_h
        else:
            n_steps = 10000
            total_arc = self.body_vel * dt_anim * n_steps
            pad = L + 0.15
            env_lim = total_arc + pad
            x_min, x_max = -env_lim, env_lim
            y_min, y_max = -env_lim / 2, env_lim / 2

        self.env = PyPlot()
        self.env.launch(limits=[
            x_min, x_max,
            y_min, y_max,
            -0.15, 0.10
        ])
        self.ax = self.env.fig.axes[0]

        if self.topdown:
            self.ax.view_init(elev=90, azim=-90)

        if floor_plan is not None:
            h, w = floor_plan.shape[:2]
            world_w = w * scale
            world_h = h * scale

            xs = np.linspace(0, world_w, w)
            ys = np.linspace(0, world_h, h)
            X, Y = np.meshgrid(xs, ys)

            # Flip vertically: image row 0 is top, but plot Y increases upward
            fp_display = np.flipud(floor_plan).astype(float)
            fp_norm = fp_display / fp_display.max()

            # Downsample for rendering performance
            step = 2
            self.ax.plot_surface(
                X[::step, ::step], Y[::step, ::step], np.zeros_like(X[::step, ::step]),
                facecolors=plt.cm.gray(fp_norm[::step, ::step]),
                rstride=1, cstride=1,
                shade=False, zorder=0
            )

        if self.path is not None:
            # path columns: [row, col] --> world: x = col*scale, y = row*scale
            path_x = self.path[:, 0] * scale
            path_y = self.path[:, 1] * scale
            path_z = np.full_like(path_x, 0.05)  # 5mm above floor so it's visible over plot_surface
            self.ax.plot3D(path_x, path_y, path_z, color='red', linewidth=2)

        T_wb = sm.SE3(0, 0, 0)

        for i, leg_robot in enumerate(self.legs):
            leg_robot.base = T_wb * self.leg_offsets[i]
            leg_robot.q = np.r_[0, 0, 0]
            self.env.add(leg_robot, readonly=True, jointaxes=False, eeframe=False, shadow=False)

        self.body = Cuboid([L, 100 * mm, 30 * mm], color='b')
        self.body.base = T_wb

        self.env.add(self.body)
        self.env.step()

    def _create_leg(self) -> rt.ERobot:
        print("creating the leg")
        leg = rt.ERobot(rt.ET.Rz() * rt.ET.Rx() * rt.ET.ty(self.L1) * rt.ET.Rx() * rt.ET.tz(-self.L2))
        print("Leg: ", leg)
        return leg

    def _create_legs(self, leg: rt.ERobot) -> list:
        legs = [
            rt.ERobot(leg, name='leg0'),
            rt.ERobot(leg, name='leg1'),
            rt.ERobot(leg, name='leg2'),
            rt.ERobot(leg, name='leg3')
        ]
        return legs

    def _create_leg_offsets(self) -> list:
        leg_offsets = [
            sm.SE3(self.BODY_WIDTH / 2,  -self.BODY_WIDTH / 2, 0),
            sm.SE3(-self.BODY_LENGTH / 2, -self.BODY_WIDTH / 2, 0),
            sm.SE3(self.BODY_LENGTH / 2,   self.BODY_WIDTH / 2, 0) * sm.SE3.Rz(np.pi),
            sm.SE3(-self.BODY_LENGTH / 2,  self.BODY_WIDTH / 2, 0) * sm.SE3.Rz(np.pi),
        ]
        return leg_offsets

    def _gait(self, cycle, k, offset, flip):
        k = (k + offset) % cycle.shape[0]
        q = cycle[k, :].copy()
        if flip:
            q[0] = -q[0]
        return q

    def run(self):
        pos_x = self._goal_list[0][0]
        pos_y = self._goal_list[0][1]

        # Point toward first waypoint so there's no initial spin
        if len(self._goal_list) > 1:
            dx = self._goal_list[1][0] - pos_x
            dy = self._goal_list[1][1] - pos_y
            theta = np.arctan2(dy, dx)
        else:
            theta = 0.0

        K_p = 2.0
        i_goal = 1  # already at waypoint 0, navigate toward waypoint 1
        i = 0

        while True:
            if not plt.fignum_exists(self.env.fig.number):
                break

            goal = self._goal_list[i_goal]
            bearing = np.arctan2(goal[1] - pos_y, goal[0] - pos_x)
            heading_error = (bearing - theta + np.pi) % (2 * np.pi) - np.pi
            turn_rad_per_step = K_p * heading_error * self._dt
            dist_to_goal = np.hypot(goal[0] - pos_x, goal[1] - pos_y)

            if dist_to_goal < self.GOAL_RADIUS:
                print(f"Goal {goal} reached at step {i}!")
                i_goal += 1
                if i_goal < len(self._goal_list):
                    print(f"Next goal is {self._goal_list[i_goal]}")

            if i_goal >= len(self._goal_list):
                print("We have reached every goal!")
                break

            # 1. Update joint angles
            self.legs[0].q = self._gait(self.qcycle, i * self._step_size,   0, False)
            self.legs[1].q = self._gait(self.qcycle, i * self._step_size, 100, False)
            self.legs[2].q = self._gait(self.qcycle, i * self._step_size, 200, True)
            self.legs[3].q = self._gait(self.qcycle, i * self._step_size, 300, True)

            # 2. Update heading
            theta += turn_rad_per_step

            # 3. Advance position — scale speed by alignment so robot turns before charging forward
            alignment = max(0.0, np.cos(heading_error))
            ds = self.body_vel * self._dt * alignment
            pos_x += ds * np.cos(theta)
            pos_y += ds * np.sin(theta)

            # 4. Build world-to-base transform
            T_wb = sm.SE3(pos_x, pos_y, 0) * sm.SE3.Rz(theta)

            for j, leg_robot in enumerate(self.legs):
                leg_robot.base = T_wb * self.leg_offsets[j]

            self.body.base = T_wb

            ax = self.ax
            cam_dist = self._cam_dist

            if self._skip <= 0:
                self.env.step(dt=self._dt)
                self.env.step(dt=self._dt)
                if self._follow_cam:
                    ax.set_xlim(pos_x - cam_dist, pos_x + cam_dist)
                    ax.set_ylim(pos_y - cam_dist, pos_y + cam_dist)
            elif i % self._skip == 0:
                self.env.step(dt=self._dt)
                if self._follow_cam:
                    ax.set_xlim(pos_x - cam_dist, pos_x + cam_dist)
                    ax.set_ylim(pos_y - cam_dist, pos_y + cam_dist)

            if self.topdown:
                ax.view_init(elev=90, azim=-90)

            i += 1

        self.env.hold()
        plt.close('all')
