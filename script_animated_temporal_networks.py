#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import shelve as sl
import numpy.random as nprm
from matplotlib.lines import Line2D
import matplotlib

from gplotclass import plot_G
from Epifunc import plot_state_evolution, get_figgrid,TimeArrays

# to improve quality of the video
plt.rcParams['figure.dpi']  = 100
plt.rcParams['savefig.dpi'] = 100


# ### Reading the network
#
# First, we read the network, aggregated in blocks of `dt=15` minutes for each day. This is equivalent to the DYN representation used in [1]

# In[2]:


# initial time of the dataset
t0 = 0
# days present in the dataset
days = list(range(0,5)) + list(range(7,12))
# hours of activity and time step (in seconds)
workinghours = 24*60*60
dt = 15*60 #dt in seconds
nblocks = int(24*60*60/dt)
nworkblocks = int(workinghours/dt)
blocks = range(0,nblocks)

file_meta = 'Data/metadata_InVS15.csv'
contact_meta = pd.read_csv(file_meta,names=['node name','class'],sep='\t').set_index('node name')

file = 'Data/tij_InVS15.csv'
daylynets = {}


for day in days:
    daylynets[day] = [ nx.Graph() for b in blocks ]

f = open(file)

for line in f:
    (t,i,j) = line.split()
    ts = (int(t)-t0)//dt; i = int(i); j = int(j)
    day = ts//nblocks
    block = ts%nblocks
    if block < nworkblocks:
        if daylynets[day][block].has_edge(i,j):
            w = daylynets[day][block].get_edge_data(i,j)['weight']
        else:
            w = 0
        daylynets[day][block].add_edge(i,j,weight = w + 20)
f.close()
G15mins =  [ daylynets[day] for day in days ]

#### wrong index for G15mins!!! starts at 00h00


# We can aggregate data even further into daily networks `G_dHET`, or an unique network for all the dataset `G_HET`


# In[5]:

import matplotlib

list_classes = np.array( contact_meta['class'].unique() )
number_categories = list_classes.size


# this can be use to assign color using a continuous colormap
#cmap = matplotlib.cm.get_cmap('inferno')
#ClassColor = {c: cmap(val/number_categories) for val,c in enumerate(list_classes)}

ClassColor = {c: 'C{}'.format(val) for val,c in enumerate(list_classes)}



nodes_color = np.array( [ ClassColor[c] for c in contact_meta['class']] )

Gplot = plot_G(contact_meta)

tmax = 12
dt_day = dt/(24*60*60)
time = np.arange(0,tmax+dt_day,dt_day)
TO =  TimeArrays(dt_sec=dt)
workingtime, block_day, data_day = TO.times(time)


# To generate the video we define the function `update(num)` which updates the figure for each timestep `num`. Outside the function (and outside the axes `ax`, which is cleared for each frame) we can put other objects, such as the legend associated with the figure.

# In[6]:

t_initial = 0 # day 43, 7h30
t_final =  12           # day 44 0h00

frames = range( int(t_initial*nblocks), int(t_final*nblocks) )

fig,ax = plt.subplots(1,1,figsize=(6,6))


#graph without edges for times without activity
G0 =nx.Graph()
G0.add_nodes_from( np.arange( contact_meta.shape[0] )  )

def update(num):
    # num is the current time step
    ax.clear()
    TO.format_time(ax,time[num],workingtime[num],px=-8,py=-6, show_activity=False)
    if workingtime[num]:
        ts =  block_day[num]
        day = data_day[num]
        G = G15mins[day][ts]
        weight = np.array( [ G[u][v]['weight'] for u,v in G.edges ] )
        # the number 1/600 was manually chosen to give an aestically coherent width to the edges
        # different networks would need a different number
        nweight = weight/600
        Gplot.plotting_edges(G,width=nweight)
    else:
        G = G0

    Gplot.plotting_nodes(G,s=10,c=nodes_color,facecolors='none',alpha=0.8)
    Gplot.plotting_group_label()
    ax.axis('off')
    ax.set_aspect('equal')

ani = matplotlib.animation.FuncAnimation(fig, update,
                                         frames=frames, interval=1000)

writervideo = matplotlib.animation.FFMpegWriter(fps=9, bitrate=1024)
ani.save("video_InVS15_agg15min_fps9.mp4", writer=writervideo)

