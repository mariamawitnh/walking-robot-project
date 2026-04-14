""" Serves as the file for abstracting setting everything up """
import roboticstoolbox as rt
import numpy as np
import matplotlib.pyplot as plt
import spatialmath as sm
from spatialgeometry import Cuboid  # seful for Cuboid, idk
from roboticstoolbox.backends.PyPlot import PyPlot


class WalkingRobot:
    """ A class for reusing the robot things, units in metres (usually) """
    GOAL_RADIUS = 0.005  # 5mm
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
                 scale=1.0,
                 topdown=False,
                 cam_dist=0.4
                 ):
        """ For some reason, ``cycle_time`` needs to be a float. It can never be an int """
        if path is None:
            self._goal_list = [(p[0]*scale, p[1]*scale) for p in goal_list]  # excpects the transform ig
            self.path = None
        else:
            # ndarray ig, actually test for this or something?
            self._goal_list = [(p[1]*scale, p[0]*scale) for p in path]
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
        """ Init the class itself, prepare everything to run later """
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

        cycle_time = cycle_time
        stroke_len = (xf - xb) * mm
        self.body_vel = stroke_len / cycle_time

        W = 100 * mm
        L = 200 * mm

        n_steps = 10000  # deprecated idk
        total_arc = self.body_vel * dt_anim * n_steps  # approx 2.0 m
        pad = L + 0.15

        env_lim = total_arc + pad

        self.env = PyPlot()
        self.env.launch(limits=[
            -env_lim,  env_lim,   # x
            -env_lim / 2,  env_lim / 2,   # y
            -0.15,     0.10       # z
        ])
        self.ax = self.env.fig.axes[0]
        if self.topdown:
            self.ax.view_init(elev=90, azim=-90)

        if floor_plan is not None:
            fp = floor_plan[::1][::1]
            h, w = fp.shape[:2]
            xs = np.linspace(-env_lim, env_lim, w)
            ys = np.linspace(-env_lim / 2, env_lim / 2, h)
            X, Y = np.meshgrid(xs, ys)
            self.ax.plot_surface(X, Y, np.zeros_like(X),
                                 facecolors=plt.cm.gray(fp / fp.max()),
                                 rstride=1, cstride=1, shade=False, zorder=0)

        if self.path is not None:
            self.ax.plot(self.path[0], self.path[1], "r", linewidth=2)
        T_wb = sm.SE3(0, 0, 0)

        for i, leg_robot in enumerate(self.legs):
            leg_robot.base = T_wb * self.leg_offsets[i]
            leg_robot.q = np.r_[0, 0, 0]
            self.env.add(leg_robot, readonly=True, jointaxes=False, eeframe=False, shadow=False)

        self.body = Cuboid([L, W, 30 * mm], color='b')
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
            sm.SE3(self.BODY_WIDTH/2,  -self.BODY_WIDTH/2, 0),
            sm.SE3(-self.BODY_LENGTH/2,  -self.BODY_WIDTH/2, 0),
            sm.SE3(self.BODY_LENGTH/2,   self.BODY_WIDTH/2, 0) * sm.SE3.Rz(np.pi),
            sm.SE3(-self.BODY_LENGTH/2,   self.BODY_WIDTH/2, 0) * sm.SE3.Rz(np.pi),
        ]
        return leg_offsets

    def _gait(self, cycle, k, offset, flip):
        k = (k + offset) % cycle.shape[0]
        q = cycle[k, :].copy()
        if flip:
            q[0] = -q[0]
        return q

    def run(self):
        pos_x = 0.0
        pos_y = 0.0
        theta = 0.0

        # running loop
        K_p = 2.0
        i_goal = 0
        i = 0
        while True:
            if not plt.fignum_exists(self.env.fig.number):
                break

            goal = self._goal_list[i_goal]
            bearing = np.arctan2(goal[1] - pos_y, goal[0] - pos_x)
            heading_error = (bearing - theta + np.pi) % (2*np.pi) - np.pi
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

            # 2. Update heading (accumulate turn each step)
            theta += turn_rad_per_step

            # 3. Advance position along current heading direction
            ds = self.body_vel * self._dt
            pos_x += ds * np.cos(theta)
            pos_y += ds * np.sin(theta)

            # 4. Build world-to-base transform:  T_wb = Translation * Rz(theta)
            #    This is the pose of the robot body in world coordinates.
            #    Leg bases are then:  T_wb * leg_offset_in_body_frame  (Eq. 7.3)
            T_wb = sm.SE3(pos_x, pos_y, 0) * sm.SE3.Rz(theta)

            for j, leg_robot in enumerate(self.legs):
                leg_robot.base = T_wb * self.leg_offsets[j]

            self.body.base = T_wb
            ax = self.ax
            cam_dist = self._cam_dist
            if self._skip <= 0:
                self.env.step(dt=self._dt)
                self.env.step(dt=self._dt)
                ax.set_xlim(pos_x - cam_dist, pos_x + cam_dist)
                ax.set_ylim(pos_y - cam_dist, pos_y + cam_dist)
            elif i % self._skip == 0:
                self.env.step(dt=self._dt)
                ax.set_xlim(pos_x - cam_dist, pos_x + cam_dist)
                ax.set_ylim(pos_y - cam_dist, pos_y + cam_dist)

            if self.topdown:
                ax.view_init(elev=90, azim=-90)

            i += 1

        self.env.hold()
        plt.close('all')
