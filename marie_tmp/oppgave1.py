""" Denne skal løse oppgave 1 """
import walkingrobot1


# Punkter vi skal nå~ (enhet m)
# Først A-B, så A-C, så til D
A = (0, 0, 0)
B = (10, 0, 0)
C = (0, 0, -1)  # \pm 10
D = (10, 0, 1)  # \pm 10

goal_list = [A, B, A, C, A, D]

robot = walkingrobot1.WalkingRobot(goal_list, anim_skip_every=100)
robot.run()
