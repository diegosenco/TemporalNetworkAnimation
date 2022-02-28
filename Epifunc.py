#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:43:30 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def evolution_list(hist, states_list):
    Xt = []
    for nstates in hist:
        n = [int(np.count_nonzero(nstates == s)) for s in states_list]
        Xt.append(n)
    return np.array(Xt)


def reduced_evolution_list(state_history, dt):
    states_list = ['S', 'E', 'Ip', 'Ic', 'Isc', 'R']
    X = evolution_list(state_history, states_list)
    X = pd.DataFrame(X, columns=states_list)
    X['I'] = X[['Ip', 'Ic', 'Isc']].sum(axis=1)
    X['size'] = 232 - X['S']
    X['time'] = np.arange(X.shape[0]) * dt
    return X[['time', 'S', 'E', 'I', 'R', 'size']]


def plot_state_evolution(state_history, dt, ax, LineColors):
    dfs = reduced_evolution_list(state_history, dt)
    dfm = dfs.melt('time', var_name='state', value_name='n')
    sns.lineplot(data=dfm, x='time', y='n', hue='state',
                 hue_order=LineColors.keys(),
                 palette=LineColors.values(),
                 ax=ax)
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_xlabel('time [days]')


def get_figgrid(size=(8, 7)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    p1 = np.array([0.05, 0.3, 0.9, 0.6])
    p2 = np.array([0.45, 0.1, 0.4, 0.16])
    ax1.set_position(p1)
    ax2.set_position(p2)
    return fig, [ax1, ax2]


class TimeArrays():
    def __init__(
            self,
            week_days=[ 0,1, 2, 3, 4],
            data_days=10,
            initial_hour=8,
            final_hour=20,
            dt_sec=15 * 60 ):
        self.list_week_days = week_days
        self.ndays = data_days

        self.sec0 = initial_hour * 3600
        self.secf = final_hour * 3600
        self.dt_sec = dt_sec
        self.dayDict = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday',
        }

    def times(self, t):
        dt = t[1] - t[0]
        day = np.array(t % 7, int)
        second = np.array(((t % 1) * (3600 * 24)).round(), int)
        cond_day = np.logical_or.reduce(
            [day == d for d in self.list_week_days])
        cond_sec = (second >= self.sec0) * (second <= self.secf)

        timestep = np.array((second / self.dt_sec).round(), int)
        day2 = np.array(dt * (np.cumsum(cond_day) - 1) +
                        t[0], int) % self.ndays
        day2[~cond_day] = day2[~cond_day] * 0
        return cond_day * cond_sec, timestep, day2

    def format_time(self, ax, t, working, px=-7, py=-7):
        weekday = self.dayDict[int(t % 7)]
        hourmins = ((24 * t) % 24)
        hour = (int(24 * t) % 24)
        mins = str(round(60 * (hourmins % 1)) % 60 + 100)[1:]
        ax.text(
            px,
            py,
            '$t= {:.2f}$ [days] \n{} {}h{}  \nactivity time: {}'.format(
                t,
                weekday,
                hour,
                mins,
                working),
            fontsize=12)


