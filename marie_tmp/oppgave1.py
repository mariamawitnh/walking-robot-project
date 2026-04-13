""" Denne skal løse oppgave 1 """
import walkingrobot


# Punkter vi skal nå~ (enhet cm)
# er noe feil med enhetene her vs enhetene i den andre filen btw
# Først A-B, så A-C, så til D
A = (0, 0, 0)
B = (100, 0, 0)
C = (0, 0, -10)  # \pm 10
D = (100, 0, 10)  # \pm 10

goal_list = [A, B, A, C, A, D]

robot = walkingrobot.WalkingRobot(goal_list, anim_skip_every=1000)
robot.run()
