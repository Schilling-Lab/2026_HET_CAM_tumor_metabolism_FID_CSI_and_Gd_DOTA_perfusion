from abc import ABC, abstractmethod

import ipywidgets as widgets


class PlotterABC(ABC):
    def __init__(self):
        self.setup_figure()
        self.setup_interactive()
        self.show()

    @abstractmethod
    def setup_figure(self) -> None:
        """Sets up matplotlib figure and axes.

        Use plt.ioff() here to create the figure!
            import matplotlib.pyplot as plt

            with plt.ioff():
                self.fig, self.ax = plt.subplots()

        Creating the figure inside the 'with' statement ensures the plot is only
        rendered when we explicitly call plt.show()
        """

    @abstractmethod
    def update(self) -> None:
        """Plot-Update function used by self.setup_interactive()."""

    @abstractmethod
    def setup_interactive(self) -> None:
        """Sets up the interactive ipywidgets part."""

    @abstractmethod
    def show(self) -> None:
        """Command to render the figure."""

    def apply_min_max_to_range_sliders(self, min_val, max_val, range_slider):
        """Handels possible conflict of min > max or max < min when changing the max and min value of an ipywidgets Rangslider."""

        # sanity check
        if type(range_slider) is widgets.widget_float.FloatRangeSlider:
            if not isinstance(min_val, float) or not isinstance(max_val, float):
                raise Exception(
                    "min_val of type "
                    + str(type(min_val))
                    + " or max_val of type "
                    + str(type(max_val))
                    + " does not match FloatRangeSlider."
                )

        elif type(range_slider) is widgets.widget_float.IntRangeSlider:
            if not isinstance(min_val, int) or not isinstance(max_val, int):
                raise Exception(
                    "min_val of type "
                    + str(type(min_val))
                    + " or max_val of type "
                    + str(type(max_val))
                    + " does not match IntRangeSlider."
                )
        elif min_val > max_val:
            raise Exception(
                "New min value of "
                + str(min_val)
                + "larger than new max value of "
                + str(max_val)
            )
        else:
            raise Exception(
                "Slider of type "
                + str(type(range_slider))
                + " is not a FloatRangeSlider or IntRangeSlider."
            )

        # retireve current slider range
        slider_old_max = range_slider.max

        if slider_old_max < min_val:
            range_slider.max = max_val
            range_slider.min = min_val
        else:
            range_slider.min = min_val
            range_slider.max = max_val
