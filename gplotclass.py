#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:49:28 2022

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
class plot_G:
    def __init__(self, contact_meta):
        '''
        this class creates the positions for nodes belonging to
        structured populations.

        it requires only a dataframe indexed by the name of the nodes,
        and a columns 'class'.
        '''
        self.node_size = 20
        self.font_size = 10
        self.edge_alpha = 0.1

        self.nodes = np.array(contact_meta.index)
        node_class = np.array(contact_meta['class'])
        Gdict = contact_meta.copy().to_dict()
        self.Lclass, count = np.unique(
            contact_meta['class'], return_counts=True)

        rad = 1
        RAD = 0.4 * self.Lclass.size * rad
        ANGLES = np.linspace(0, 2 * np.pi, self.Lclass.size + 1)
        centers = {c: (RAD * np.cos(ea), RAD * np.sin(ea))
                   for c, ea in zip(self.Lclass, ANGLES)}

        self.pos = {}

        class_rad = {}
        for c in self.Lclass:
            #nodeclass = [u for u in nodes if nodes[u]['class'] == c]
            nodeclass = self.nodes[node_class == c]
            rad = np.sqrt(len(nodeclass)) / 6.5
            class_rad[c] = rad
            angles = np.linspace(0, 2 * np.pi, len(nodeclass) + 1)
            cx, cy = centers[c]
            p = [(rad * np.cos(ea) + cx, rad * np.sin(ea) + cy)
                 for ea in angles]
            self.pos.update(dict(zip(nodeclass, p)))
        self.posarray = np.array([self.pos[n] for n in self.nodes])

        self.centers_text = {c: (
            (1.15 * RAD + class_rad[c]) * np.cos(ea),
            (1.15 * RAD + class_rad[c]) * np.sin(ea)
        )
            for c, ea in zip(self.Lclass, ANGLES)}

    def plotting(self, G, node_color='C0'):
        for c in self.Lclass:
            plt.text(*self.centers_text[c], c)
        nx.draw_networkx_nodes(G, self.pos, node_size=self.node_size,
                               node_color=node_color)
        nx.draw_networkx_edges(G, self.pos, width=1, alpha=self.edge_alpha)
        plt.axis('off')

    def plotting_group_label(self):
        for c in self.Lclass:
            plt.text(*self.centers_text[c], c,
                     fontsize=self.font_size,
                     va='center', ha='center')

    def plotting_nodes(self, G, nodes_to_show=None, **kargs):
        if not isinstance(nodes_to_show, type(None)):
            x, y = self.posarray[:,
                                 0][nodes_to_show], self.posarray[:, 1][nodes_to_show]
        else:
            x, y = self.posarray[:, 0], self.posarray[:, 1]
        plt.scatter(x, y, **kargs)

    def plotting_edges_old(self, G, width=1, alpha=0.5):
        nx.draw_networkx_edges(G, self.pos, width=width, alpha=alpha)

    def legend(self):
        plt.legend(handles=self.handles, loc=(1.1, 0.01))

    def plotting_edges(self, G, width=1, alpha=0.5, color='k'):
        for i, (u, v) in enumerate(G.edges):
            e_color = color[i] if isinstance(color, list) else color
            e_alpha = alpha[i] if hasattr(alpha, '__iter__') else alpha
            e_width = width[i] if hasattr(width, '__iter__') else width
            X = np.array([self.pos[u], self.pos[v]])
            plt.plot(X[:, 0], X[:, 1], color=e_color,
                     alpha=e_alpha, lw=e_width)
