#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  line plot
   Email : autuanliu@163.com
   Date：2018/3/30
"""
from .vis_imports import *


def line(*data, x_label='', y_label='', title='', **kwargs):
    """plot line
    """
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(*data, lw=2, color='#539caf', alpha=1, **kwargs)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
