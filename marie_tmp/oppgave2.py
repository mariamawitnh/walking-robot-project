""" kreft suger baller """
import roboticstoolbox as rt
import matplotlib.pyplot as plt
import random
# import walkingrobot
# from roboticstoolbox.backends.PyPlot import PyPlot

# følg fra 5.4 og ut i boka ellerno
house = rt.rtb_load_matfile("data/house.mat")

floorplan = house["floorplan"]
places = house["places"]


def generate_random_path_plot(i):

    # generate two unique points
    while True:
        start = places[random.choice(list(places))]
        end = places[random.choice(list(places))]

        break
        if start.all() != end.all():
            break

    # generate path
    dx = rt.DistanceTransformPlanner(occgrid=floorplan)
    dx.plan(goal=end)
    path = dx.query(start=start)

    fig, ax = plt.subplots()
    ax.imshow(floorplan, cmap="gray")

    pathT = path.T  # path transpose so it becomes (x, y)
    ax.plot(pathT[0], pathT[1], "r", linewidth=2)

    # plot start and end points
    ax.plot(start[0], start[1], 'go')   # green start
    ax.plot(end[0], end[1], 'bo')     # blue goal

    plt.show()


for i in range(5):
    generate_random_path_plot(i)
"""
scale = 0.01
goal_list = [(p[1]*scale, p[0]*scale) for p in path]
robot = walkingrobot.WalkingRobot(goal_list)
robot.run()
"""
