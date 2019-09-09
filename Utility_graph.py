import numpy as np
import os
import matplotlib.pyplot as plt
import scanner
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

def utility(x):
    return np.array(x, dtype=np.utility64)

p1 = figure(x_axis_type="Work Class", title="The values of attribute work class")
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Work Class'
p1.yaxis.axis_label = 'Utility'

p1.line(utility(AAPL['']), AAPL['adj_close'], color='#A6CEE3', legend='AAPL')
p1.line(utility(GOOG['work class']), GOOG['adj_close'], color='#B2DF8A', legend='GOOG')
p1.line(utility(IBM['work class']), IBM['adj_close'], color='#33A02C', legend='IBM')
p1.line(utility(MSFT['work class']), MSFT['adj_close'], color='#FB9A99', legend='MSFT')
p1.legend.location = "top_left"

aapl = np.array(AAPL['adj_close'])
aapl_workclasses = np.array(AAPL['workclass'], dtype=np.Utility64)

window_size = 30
window = np.ones(window_size)/float(window_size)
aapl_avg = np.convolve(aapl, window, 'same')

p2 = figure(x_axis_type="Utility", title="AAPL One-Month Average")
p2.grid.grid_line_alpha = 0
p2.xaxis.axis_label = 'Work Class'
p2.yaxis.axis_label = 'Utility'
p2.ygrid.band_fill_color = "Blue"
p2.ygrid.band_fill_alpha = 0.1

p2.circle(aapl_dates, aapl, size=4, legend='close',
          color='darkgrey', alpha=0.2)

p2.line(aapl_dates, aapl_avg, legend='avg', color='navy')
p2.legend.location = "top_left"

output_file("utility.html", title="utility.py example")

show(gridplot([[p1,p2]], plot_width=400, plot_height=400))  # open a browser