import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("durations.csv")


def plot_based_on_group(grouping="npoints", plotting="time", xlabel="time", ylabel="npoints", title="Title"):
    plt.figure(figsize=(10, 6))

    for group_value, group in df.groupby(grouping):
        plt.plot(group[plotting], group[grouping], label=f'{grouping}: {group_value}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

avg_time = df.groupby("npoints")["time"].mean()
avg_fail = df.groupby("npoints")["amount_of_fails"].mean()
print(f"Average time: {avg_time}")
print(f"Average fail: {avg_fail}")

# plot time
plot_based_on_group(ylabel="npoints", xlabel="time (s)", title="How much time per group of npoints")

# plot amount_of_fails
plot_based_on_group(plotting="amount_of_fails", xlabel="times", title="How many failed per group of npoints")
