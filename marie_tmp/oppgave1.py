""" Denne skal løse oppgave 1 """
import walkingrobot


# Punkter vi skal nå~ (enhet cm)
# Først A-B, så A-C, så til D
A = (0, 0, 0)
B = (100, 0, 0)
C = (0, 0, -10)  # \pm 10
D = (100, 0, 10)  # \pm 10

goal_list = [[A, B], [A, C], [A, D]]

robot = walkingrobot.WalkingRobot(goal_list)
