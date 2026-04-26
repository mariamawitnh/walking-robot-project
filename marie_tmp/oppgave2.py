""" Oppgave 2 """
import roboticstoolbox as rt
import matplotlib.pyplot as plt
import random
import walkingrobot2 as walkingrobot2
import numpy as np

# følg fra 5.4 og ut i boka ellerno
house = rt.rtb_load_matfile("data/house.mat")

floorplan = house["floorplan"]
places = house["places"]


def generate_random_path_plot(i, npoints=400, plot_prm=False):

    # generate two unique points
    while True:
        try:
            place1 = random.choice(list(places))
            start = places[place1]
            place2 = random.choice(list(places))
            end = places[place2]

            # if cross of the two vectors equal zero, then try try again
            if np.cross(start, end) == 0:
                continue

            # generate path
            print(place1, place2, start, end)
            prm = rt.PRMPlanner(occgrid=floorplan, npoints=npoints)
            prm.plan()
            path = prm.query(start=start, goal=end)

            if path is not None and len(path) != 0:
                break

            npoints += 100
            print("Increased npoints as the previous value was not enough")
        except Exception:
            print("failed to find a path, trying again")

    if plot_prm:
        fig, ax = plt.subplots()
        ax.imshow(floorplan, cmap="gray")

        # plot the random nodes and the path
        prm.plot(background=True)

        pathT = path.T  # path transpose so it becomes (x, y)
        ax.plot(pathT[0], pathT[1], "white", linewidth=2, label="path")

        ax.legend()
        ax.set_title(f"PRM Plot (npoints={npoints})")

        plt.show()

    # goal_list = [(p[0], p[1]) for p in path]
    robot = walkingrobot2.WalkingRobot(
        floor_plan=floorplan, anim_skip_every=100, path=path)
    robot.run()


for i in range(5):
    generate_random_path_plot(i, plot_prm=False)
