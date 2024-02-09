import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot(var, y_label=" ", x_label="Iterations", title="", save_figure=False, filename=None):
    """
    Creates plot of var (y-axis: var, x-axis: [0, ..., len(var)] )
    ------
    Inputs:
        :param var: variable to be plotted
        :param y_label: label on y-axis
        :param x_label: label on x-axis
        :param title: title of the plot
        :param save_figure: if True, then saves figure in a file with name "filename"
        :param filename: the name of the file, where the plot is saved
    """

    plt.plot(np.arange(np.shape(var)[0]), var, marker="o", linestyle='-', linewidth=2)

    plt.yscale('log')
    plt.grid(True)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()

    if save_figure:
        if filename is None:
            raise Exception("Error: please provide filename for saving the file!")
        plt.savefig(filename + ".png")
    plt.show()

def plot_multiple_var(multiple_vars, labels, y_label=" ", x_label="Iterations", title="", save_figure=False, filename=None):
    """
    Plot multiple variables
    ------
    Inputs:
        - multiple_vars (list): list of all the variables that you want to plot
        - labels (list): list of y-axis labels for each variable that is plotted
        - x_label (str): label of x-axis
        - title (str): title of the plot
        - save_figure (bool): If True, saves figure
        - filename (str): name of the file where the plot will be saved to.
    Returns: None
    """
    colors = ['#0525f5', '#ff9429', 'green', 'red', 'purple']  # Define colors for different variables
    plt.figure(figsize=(26, 19))
    markers = itertools.cycle(('o', '*', 's', '>', 'P', 'o', '*', '>', 5, 'P'))
    for i in range(len(multiple_vars)):
        plt.plot(np.arange(np.shape(multiple_vars[i])[0]), multiple_vars[i], marker=next(markers), label=labels[i],
                 markersize=32)
    plt.yscale('log')
    plt.grid(True)
    plt.ylabel(y_label, fontsize=69)
    plt.yticks(fontsize=54)
    plt.xlabel(x_label, fontsize=69)
    plt.xticks(fontsize=54)
    plt.title(title, fontsize=67, y=1.0, pad=+25)
    plt.legend(fontsize="54", loc='upper right')

    if save_figure:
        if filename is None:
            raise Exception("Error: please provide filename for saving the file!")
        plt.savefig(filename+".pdf", format="pdf", bbox_inches="tight")

    plt.show()

def plot_2d(vars, labels, sol=None, save_figure=False, filename=None, title=''):
    """
    Plot 2D data points with a cursor showing the trajectory, a star for the specified solution,
    a circular point for the initial point

    Inputs:
        x_values (list): List of x-coordinate values.
        y_values (list): List of y-coordinate values.
        sol (tuple): Tuple of (x, y) representing the solution coordinates.
        f (callable): Input function f(x, y) that returns the flow vectors (U, V) at coordinates (x, y).
    """
    colors = ['#0525f5', '#ff9429', 'green', 'red', 'purple']  # Define colors for different variables

    fig, ax = plt.subplots()
    fig.set_figheight(11)
    fig.set_figwidth(15)
    markers = itertools.cycle(('o', '>', '*', 5 , 'P', 'o', '*', '>', 5, 'P'))
    for i in range(len(vars)):
        x_values = []
        y_values = []
        var = vars[i]
        for elem in var:
            x_values.append(elem[0])
            y_values.append(elem[1])

        color = colors[i % len(colors)]
        ax.plot(x_values, y_values, color=color, marker=next(markers), markersize=14, label=labels[i])
        _, = ax.plot([], [], color=color, linestyle='--')  # Cursor line
        _, = ax.plot(x_values[0], y_values[0], 'o', color='grey', markersize=16)  # Initial point

        # Plot dashed lines between consecutive points
        for i in range(len(x_values) - 1):
            ax.plot([x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]], '--', color=color)

        if sol is not None:
            star = ax.plot(*sol, 'k*', markersize=18, color="#FF2511")  # Solution point with black marker

    # Add legends for the different variables
    legend_elements = [plt.Line2D([0], [0], marker='o', color='grey', label='Initial Point', markersize=22)]
    markers = ['o', '>', '*', 5 , 'P', 'o', '*', '>', 5, 'P']
    if sol is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='red', label='Solution Point', markersize=22))

    for i, label in enumerate(labels):
        legend_elements.append(plt.Line2D([0], [0], marker=markers[i], color=colors[i % len(colors)], markerfacecolor=colors[i % len(colors)],
                                      label=label, markersize=22))

    plt.legend(handles=legend_elements, fontsize="26")
    plt.xlabel('x-axis', fontsize=44)
    plt.xticks(fontsize= 34)
    plt.ylabel('y-axis', fontsize=44)
    plt.yticks(fontsize=34)
    plt.title(title, fontsize=49)
    plt.grid(True)

    if save_figure:
        if filename is None:
            print("Error: please provide filename for saving the file!")
            return
        plt.savefig(filename + ".pdf")
    plt.show()