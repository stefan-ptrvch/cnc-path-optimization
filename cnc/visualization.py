import numpy as np
from bokeh.layouts import column, row, widgetbox
from bokeh.models import CustomJS, Slider
from bokeh.models.widgets.buttons import Button
from bokeh.plotting import figure, output_file, save, ColumnDataSource
from bokeh.palettes import Category20c_20, Category10_4
from bokeh.models.markers import Triangle


class Visualizer():
    """
    Visualizes results of GA optimization of CNC shortest path problem.
    """

    def __init__(self, result, initial):
        self.result = result
        self.initial = initial

    def populate_plot(self, plot, data):
        """
        Draws one visualization on a plot.
        """

        # Determine which type of line gets which color
        color_map = {
                'REF': Category20c_20[16],
                'SCRIBE_LINE': Category20c_20[0],
                'SCRIBE_LINE1': Category20c_20[0],
                'SCRIBE_LINE2': Category20c_20[1],
                'SCRIBE_LINE3': Category20c_20[2],
                'SCRIBE_LINE4': Category20c_20[3],
                'BUSBAR_LINE': Category20c_20[4],
                'BUSBAR_LINE1': Category20c_20[4],
                'BUSBAR_LINE2': Category20c_20[5],
                'BUSBAR_LINE3': Category20c_20[6],
                'BUSBAR_LINE4': Category20c_20[7],
                'EDGEDEL_LINE': Category20c_20[8],
                'EDGEDEL_LINE1': Category20c_20[8],
                'EDGEDEL_LINE2': Category20c_20[9],
                'EDGEDEL_LINE3': Category20c_20[10],
                'EDGEDEL_LINE4': Category20c_20[11]
                }
        # Color of the non cutting line
        red = Category10_4[3]
        radius = 13
        line_width = 3

        scatter_points = {}
        for line in data:
            # Sort scatter points
            if line[0] not in scatter_points:
                scatter_points[line[0]] = {
                        'x': [line[1][0, 0], line[1][1, 0]],
                        'y': [line[1][0, 1], line[1][1, 1]]
                        }
            else:
                scatter_points[line[0]]['x'].append(line[1][0, 0])
                scatter_points[line[0]]['x'].append(line[1][1, 0])
                scatter_points[line[0]]['y'].append(line[1][0, 1])
                scatter_points[line[0]]['y'].append(line[1][1, 1])

            # Cutting line
            plot.line(
                    [line[1][0, 0], line[1][1, 0]],
                    [line[1][0, 1], line[1][1, 1]],
                    color=color_map[line[0]],
                    line_width=line_width
                    )

        # Add a scatter plot for every group
        for group_name, group in scatter_points.items():
            plot.scatter(
                    group['x'],
                    group['y'],
                    color=color_map[group_name],
                    radius=radius,
                    legend=group_name
                    )

        # Add travel lines
        for line in range(len(data) - 1):
            plot.line(
                    [
                        data[line][1][1][0],
                        data[line + 1][1][0][0]],
                    [
                        data[line][1][1][1],
                        data[line + 1][1][0][1]
                        ],
                    color='black',
                    legend='Non Cutting'
                    )

        return plot


    def split_line(self, start, end, increment):
        """
        Function that generates desired number of points between two points.
        """

        num_splits = int(np.linalg.norm(end - start)/increment)
        return (
                np.linspace(start[0], end[0], num_splits),
                np.linspace(start[1], end[1], num_splits)
                )


    def generate_tool_path(self, data, step_size):
        """
        Function that generates the path for the cutting tool, used for
        visualization.
        """

        lines_x = np.ndarray((0))
        lines_y = np.ndarray((0))
        for line_number in range(len(data) - 1):

            line = data[line_number][1]
            next_line = data[line_number + 1][1]

            # Cutting line
            line_x, line_y = self.split_line(line[0], line[1], step_size)
            lines_x = np.hstack((lines_x, line_x))
            lines_y = np.hstack((lines_y, line_y))

            # Non cutting line
            line_x, line_y = self.split_line(line[1], next_line[0], step_size)
            lines_x = np.hstack((lines_x, line_x))
            lines_y = np.hstack((lines_y, line_y))

        # Add the last line (cutting line)
        line_x, line_y = self.split_line(
                data[-1][1][0],
                data[-1][1][1],
                step_size
                )
        lines_x = np.hstack((lines_x, line_x))
        lines_y = np.hstack((lines_y, line_y))

        return lines_x, lines_y


    def visualize_solution(self):
        """
        Draw the plot of the best solution.
        """

        # Tools that will be displayed on the plots
        tools = "pan,wheel_zoom,reset"

        # Plot displaying the optimized path
        result_plot = figure(
                plot_width=1000,
                plot_height=500,
                tools=tools,
                active_scroll='wheel_zoom')
        result_plot.title.text = "Optimized Path"

        # Plot displaying the non optimized path
        initial_plot = figure(
                plot_width=1000,
                plot_height=500,
                tools=tools,
                active_scroll='wheel_zoom')
        initial_plot.title.text = "Initial Path"

        # Add the data to the result plot
        result_plot = self.populate_plot(result_plot, self.result)
        result_plot.legend.location = "bottom_right"

        # Add the data to the initial plot
        initial_plot = self.populate_plot(initial_plot, self.initial)
        initial_plot.legend.location = "bottom_right"

        # Add cutting tool to plots
        # Generate the points on which the triangle should move on
        result_lines_x, result_lines_y = self.generate_tool_path(self.result, 1)
        initial_lines_x, initial_lines_y = self.generate_tool_path(self.initial, 1)

        # Add cutting tool triangle to optimized path
        result_triangle_position = ColumnDataSource(
                data=dict(
                    x=[result_lines_x[0]],
                    y=[result_lines_y[0]]
                    ))
        result_triangle = Triangle(
                x='x', y='y', line_color=Category10_4[3], line_width=3,
                size=20, fill_alpha=0
                )
        result_plot.add_glyph(result_triangle_position, result_triangle)

        # Add cutting tool triangle to initial path
        initial_triangle_position = ColumnDataSource(
                data=dict(
                    x=[initial_lines_x[0]],
                    y=[initial_lines_y[0]]
                    ))
        initial_triangle = Triangle(
                x='x', y='y', line_color=Category10_4[3], line_width=3,
                size=20, fill_alpha=0
                )
        initial_plot.add_glyph(initial_triangle_position, initial_triangle)

        # Add button to start moving the triangle
        button = Button(label='Start')
        result_num_steps = result_lines_x.shape[0]
        initial_num_steps = initial_lines_x.shape[0]
        num_steps = max(result_num_steps, initial_num_steps)

        # JavaScript callback which will be called once the button is pressed
        callback = CustomJS(args=dict(
            result_triangle_position=result_triangle_position,
            result_lines_x=result_lines_x,
            result_lines_y=result_lines_y,
            result_num_steps=result_num_steps,
            initial_triangle_position=initial_triangle_position,
            initial_lines_x=initial_lines_x,
            initial_lines_y=initial_lines_y,
            initial_num_steps=initial_num_steps,
            num_steps=num_steps
            ),
        code="""
            // Animate optimal path plot
            for(let i = 0; i < num_steps; i += 50) {
                setTimeout(function() {
                    if (i < result_num_steps) {
                        result_triangle_position.data['x'][0] = result_lines_x[i]
                        result_triangle_position.data['y'][0] = result_lines_y[i]
                    }

                    if (i < initial_num_steps) {
                        initial_triangle_position.data['x'][0] = initial_lines_x[i]
                        initial_triangle_position.data['y'][0] = initial_lines_y[i]
                    }

                    result_triangle_position.change.emit()
                    initial_triangle_position.change.emit()

                }, i)
            }
        """)
        # Add callback function to button, which starts the whole animation
        button.js_on_click(callback)

        # Save the plot
        result_plot = row([result_plot, button])
        plot = column([result_plot, initial_plot])
        output_file("result.html", title="CNC Path Optimization")
        save(plot)
