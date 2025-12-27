import matplotlib

FIG_SIZE = (15, 6)
WINDOW_SIZE = 30


def savefig(plt: matplotlib.figure.Figure, basename: str) -> None:
    plt.savefig(f"generated/{basename}.png")
    matplotlib.pyplot.close()
