import numpy as np

from traits.api import HasStrictTraits, Instance, List, Array
from chaco.api import Plot, ArrayPlotData
from enable.base_tool import BaseTool


class SelectionPlot(HasStrictTraits):

    image = Array()

    plot = Instance(Plot)

    def _plot_default(self):

        # Create a plot data obect and give it this data
        pd = ArrayPlotData()
        pd.set_data("imagedata", self.image)

        # Create the plot
        plot = Plot(pd)
        img_plot = plot.img_plot("imagedata")

        # Tweak some of the plot properties
        plot.padding = 0

        # Attach some tools to the plot
        plot.tools.append(ClickTool(plot))
        return plot


class ClickTool(BaseTool):

    vertices = List()

    def normal_left_down(self, event):
        self.vertices.append(self.component.map_data((event.x, event.y)))
        print self.vertices[-1]
        np.save('vertices.npy', np.array(self.vertices))


import enaml
from enaml.qt.qt_application import QtApplication


def main():
    import sys
    from scipy.ndimage import imread
    with enaml.imports():
        from selection_plot_view import Main

    app = QtApplication()

    img_array = imread(sys.argv[1], flatten=True)

    selection_plot = SelectionPlot(image=np.flipud(img_array))
    view = Main(plot=selection_plot.plot)
    view.show()

    # Start the application event loop
    app.start()

if __name__ == "__main__":
    main()
