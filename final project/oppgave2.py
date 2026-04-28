""" dette løser oppgave 2 """
import roboticstoolbox as rt
import matplotlib.pyplot as plt
# import walkingrobot
# from roboticstoolbox.backends.PyPlot import PyPlot


# følg fra 5.4 og ut i boka ellerno
house = rt.rtb_load_matfile("data/house.mat")

floorplan = house["floorplan"]
places = house["places"]

start = places["kitchen"]
end = places["br3"]

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


"""
plt.imshow(floorplan, cmap="gray")

# Plot room locations
for name, coord in places.items():
    x, y = coord
    plt.scatter(x, y, c="red")
    plt.text(x + 5, y + 5, name, color="red")
"""
plt.show()
"""
scale = 0.01
goal_list = [(p[1]*scale, p[0]*scale) for p in path]
robot = walkingrobot.WalkingRobot(goal_list)
robot.run()
"""
