from bokeh.layouts import row, widgetbox
from bokeh.models import CustomJS, Slider
from bokeh.plotting import figure, output_file, save, ColumnDataSource
from bokeh.palettes import Category20


class Visualizer():
    """
    Visualizes results of GA optimization of CNC shortest path problem.
    """

    def __init__(self, optimizer):
        self.opt = optimizer


    def visualize_solution(self):
        """
        Draw the plot of the best solution.
        """

        # For every line, draw the start and end position, and connect them
        # with a line
        coordinates = [
                self.opt.nodes[node] for node in self.opt.best_result['solution']
                ]

        radius = 10
        plot = figure(plot_width=800, plot_height=800)
        for line in coordinates:
            # Startpoint
            plot.scatter(line[0][0], line[0][1], color='blue', radius=radius,
                    fill_alpha=0.6)

            # Endpoint
            plot.scatter(line[1][0], line[1][1], color='red', radius=radius,
                    fill_alpha=0.6)

            # Cutting line
            plot.line([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                    color='blue')

        # Add travel lines
        for line in range(len(coordinates) - 1):
            plot.line(
                    [
                        coordinates[line][1][0],
                        coordinates[line + 1][0][0]],
                    [
                        coordinates[line][1][1],
                        coordinates[line + 1][0][1]
                        ],
                    color='red')

        output_file("result.html", title="Shortest Path")
        save(plot)
