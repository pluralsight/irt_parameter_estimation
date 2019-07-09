""" Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve abcd_sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.

For this to run, you will need to install bokeh with conda or pip:
    https://github.com/bokeh/bokeh
bokeh is at version 1.2.0 at the writing of this script
"""

import os

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure

from irt_parameter_estimation.util import logistic4PLabcd

a_default = 0.002
b_default = 1500
c_default = 0
d_default = 1

# Set up data
x = np.arange(0, 3001, 50)
y = logistic4PLabcd(a_default, b_default, c_default, d_default, x)
source = ColumnDataSource(data=dict(x=x, y=y))


# Set up plot
plot = figure(
    plot_height=400,
    plot_width=400,
    title="my sine wave",
    tools="crosshair,pan,reset,save,wheel_zoom",
    x_range=[0, 3000],
    y_range=[-0.2, 1.2],
)

plot.line("x", "y", source=source, line_width=3, line_alpha=0.6)


# Set up widgets
text = TextInput(title="title", value="Item Characteristic Curve")

a = Slider(
    title="a: discriminatory power", value=0.002, start=-0.006, end=0.006, step=0.0005
)
b = Slider(title="b: difficulty", value=1500.0, start=0.0, end=3000.0, step=25)
c = Slider(title="c: guessing parameter", value=0.0, start=-0.2, end=1.0, step=0.01)
d = Slider(title="d: expert error rate", value=1.0, start=0.0, end=1.0, step=0.01)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value


text.on_change("value", update_title)


def update_data(attrname, old, new):

    # Get the current slider values
    aa = a.value
    bb = b.value
    cc = c.value
    dd = d.value

    # Generate the new curve
    x = np.arange(0, 3001, 50)
    y = logistic4PLabcd(aa, bb, cc, dd, x)

    source.data = dict(x=x, y=y)


for w in [a, b, c, d]:
    w.on_change("value", update_data)


# Set up layouts and add to document
inputs = column(text, a, b, c, d)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"

if __name__ == "__main__":
    os.system(f"bokeh serve {__file__}")
