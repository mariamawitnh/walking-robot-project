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
                 goal_list: list,
                 dt_anim=0.01,
                 cycle_time=4.01,
                 anim_skip_every=3
                 ):
        """ For some reason, ``cycle_time`` needs to be a float. It can never be an int """
        self._dt = dt_anim
        self._gait_dt = 0.01
        self._step_size = max(1, round(dt_anim / self._gait_dt))

        self.leg = self._create_leg()
        self.legs = self._create_legs(self.leg)
        self.leg_offsets = self._create_leg_offsets()

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
        qcycle = sol.q

        cycle_time = cycle_time
        stroke_len = (xf - xb) * mm
        body_vel = stroke_len / cycle_time

        W = 100 * mm
        L = 200 * mm

        n_steps = 10000  # deprecated idk
        total_arc = body_vel * dt_anim * n_steps  # approx 2.0 m
        pad = L + 0.15

        env_lim = total_arc + pad

        env = PyPlot()
        env.launch(limits=[
            -env_lim,  env_lim,   # x
            -env_lim / 2,  env_lim / 2,   # y
            -0.15,     0.10       # z
        ])

        T_wb = sm.SE3(0, 0, 0)

        for i, leg_robot in enumerate(self.legs):
            leg_robot.base = T_wb * self.leg_offsets[i]
            leg_robot.q = np.r_[0, 0, 0]
            env.add(leg_robot, readonly=True, jointaxes=False, eeframe=False, shadow=False)

        body = Cuboid([L, W, 30 * mm], color='b')
        body.base = T_wb

        # probs save this to self
        env.add(body)
        env.step()

        pos_x = 0.0
        pos_y = 0.0
        theta = 0.0

        # running loop
        K_p = 2.0
        i_goal = 0
        i = 0
        while True:
            if not plt.fignum_exists(env.fig.number):
                break

            goal = goal_list[i_goal]
            bearing = np.arctan2(goal[1] - pos_y, goal[0] - pos_x)
            heading_error = (bearing - theta + np.pi) % (2*np.pi) - np.pi
            turn_rad_per_step = K_p * heading_error * dt_anim
            dist_to_goal = np.hypot(goal[0] - pos_x, goal[1] - pos_y)
            if dist_to_goal < self.GOAL_RADIUS:
                print(f"Goal {goal} reached at step {i}!")
                i_goal += 1
                print(f"Next goal is {goal_list[i_goal]}")

            if i_goal >= len(goal_list):
                print("We have reached every goal!")
                break

            # 1. Update joint angles
            self.legs[0].q = self._gait(qcycle, i * self._step_size,   0, False)
            self.legs[1].q = self._gait(qcycle, i * self._step_size, 100, False)
            self.legs[2].q = self._gait(qcycle, i * self._step_size, 200, True)
            self.legs[3].q = self._gait(qcycle, i * self._step_size, 300, True)

            # 2. Update heading (accumulate turn each step)
            theta += turn_rad_per_step

            # 3. Advance position along current heading direction
            ds = body_vel * dt_anim
            pos_x += ds * np.cos(theta)
            pos_y += ds * np.sin(theta)

            # 4. Build world-to-base transform:  T_wb = Translation * Rz(theta)
            #    This is the pose of the robot body in world coordinates.
            #    Leg bases are then:  T_wb * leg_offset_in_body_frame  (Eq. 7.3)
            T_wb = sm.SE3(pos_x, pos_y, 0) * sm.SE3.Rz(theta)

            for j, leg_robot in enumerate(self.legs):
                leg_robot.base = T_wb * self.leg_offsets[j]

            body.base = T_wb
            ax = env.fig.axes[0]
            cam_dist = 0.4
            if self._skip <= 0:
                env.step(dt=dt_anim)
                env.step(dt=dt_anim)
                ax.set_xlim(pos_x - cam_dist, pos_x + cam_dist)
                ax.set_ylim(pos_y - cam_dist, pos_y + cam_dist)
            elif i % self._skip == 0:
                env.step(dt=dt_anim)
                ax.set_xlim(pos_x - cam_dist, pos_x + cam_dist)
                ax.set_ylim(pos_y - cam_dist, pos_y + cam_dist)

            i += 1

        env.hold()
        plt.close('all')

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
        pass
