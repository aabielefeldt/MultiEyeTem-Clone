# %%
# Imports 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# %%
# Data

df = pd.read_csv('../data/csv_files/df.csv')

oris = []
oris_value = 11.25

while oris_value <= 360:
    oris.append(oris_value)
    oris_value = oris_value + 360/16

print(oris)#

unbiased_assumption = np.full(17, 0.5)

# %%
oi_means = np.array([0.4040535887804878, 
0.4545639565853659, 
0.5471473390243903, 
0.6197317753658537, 
0.35691977975609757, 
0.4477963587804878, 
0.5144225665853659, 
0.5876999751219513, 
0.3626270643902439, 
0.4171971246341464, 
0.4768843041463415, 
0.6160859075609756, 
0.3898721646341463, 
0.5435330634146341, 
0.5719072812195123, 
0.6102951121951219])



print(type(oi_means[0]))
fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')

# Plot the data with different colors
theta = np.radians(oris)
# Use colors from the color wheel map
print(theta, oi_means)
# ax.plot([theta, theta], [0, mean_bp], 'o--', c=sns.hls_palette(16)[r], alpha = 0.5, markersize=10)
ax.plot(theta, oi_means, 'k--')

# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
ax.xaxis.grid(color='pink', linestyle='-', linewidth=1)  #
plt.show()

# %%
# Set up the color wheel map
colors = plt.cm.jet(np.linspace(0, 1, len(unbiased_assumption)))

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
# Set the background color to None (transparent)
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)
# Plot the data with different colors
for i in range(len(oris)):
    theta = np.radians(oris[i])
    unbiased = np.full(17, 0.5)[i]
    ax.plot(theta, unbiased, 'o', color=colors[i], markersize=18)  # Use colors from the color wheel map
     # Add labels to the points inside the circle
    label_offset = 0.2  # Adjust this value for label position
    ax.text(theta, unbiased + label_offset, str(oris[i]), fontsize=11, ha='center', va='center', color='black')

# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
ax.xaxis.grid(color='pink', linestyle='-', linewidth=1)  #
plt.show()

# %%
# Set up the color wheel map
colors = plt.cm.jet(np.linspace(0, 1, 17))

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
# Set the background color to None (transparent)
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)

# Remove the outer circle (frame) and grid lines
ax.spines['polar'].set_visible(False)  # Remove outer black circle
ax.grid(False)  # Remove grid lines
ax.set_rticks([])  # Remove radial ticks and grid lines

# Keep the cardinal (angular) grid lines
ax.xaxis.grid(True, color='pink', linestyle='--')  # Enable grid lines for angles

# Plot the data with different colors
for i in range(len(oris)):
    theta = np.radians(oris[i])
    unbiased = np.full(17, 0.55)[i]
    ax.plot(theta, unbiased, 'o', color=colors[i], markersize=38, alpha=0.6)  # Use colors from the color wheel map

    # Add labels to the points inside the circle
    label_offset = 0.0  # Adjust this value for label position
    ax.text(theta, unbiased + label_offset, str(oris[i]), fontsize=9, ha='center', va='center', color='black')

# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
plt.savefig('../figures/colorful_dots_orientations.png')
plt.show()

# %%
# Set up the color wheel map
colors = plt.cm.jet(np.linspace(0, 1, 17))

fig = plt.figure(figsize=(12,12))

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
# Set the background color to None (transparent)
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)

# Remove the outer circle (frame) and grid lines
ax.spines['polar'].set_visible(False)  # Remove outer black circle
ax.grid(False)  # Remove grid lines
ax.set_rticks([])  # Remove radial ticks and grid lines

# Keep the cardinal (angular) grid lines
ax.xaxis.grid(True, color='pink', linestyle='--')  # Enable grid lines for angles

# Plot the data with different colors
for i in range(len(oris)):
    theta = np.radians(oris[i])
    diff = np.radians(8)
    unbiased = np.full(17, 0.4)[i]
    color = colors[i]
    ax.plot(theta, unbiased, 'o', color=color, markersize=65, alpha=0.6)  # Use colors from the color wheel map
    ax.plot((theta - diff), unbiased + label_offset+0.1, '<', markersize=35, color=color, alpha=0.5)
    ax.plot((theta + diff), unbiased + label_offset+0.1, '>', markersize=35, color=color, alpha=0.5)


    # Add labels to the points inside the circle
    label_offset = 0.0  # Adjust this value for label position
    ax.text(theta, unbiased + label_offset, str(oris[i]), fontsize=15, ha='center', va='center', color='black')
    ax.text((theta - diff), unbiased + label_offset+0.1, (str(oris[i]-8)), fontsize=9, ha='center', va='center', color='black', alpha=0.5)
    ax.text((theta + diff), unbiased + label_offset+0.1, (str(oris[i]+8)), fontsize=9, ha='center', va='center', color='black', alpha=0.5)

# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
plt.savefig('../figures/colorful_dots_orientations.png')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

colors = plt.cm.jet(np.linspace(0, 1, 17))

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='polar')

ax.set_yticklabels([])
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)

ax.spines['polar'].set_visible(False)
ax.grid(False)
ax.set_rticks([])
ax.xaxis.grid(True, color='pink', linestyle='--')

for i in range(len(oris)):
    theta = np.radians(oris[i])
    diff = np.radians(8)

    r = 0.4
    
    color = colors[i]

    # Main dot
    ax.plot(theta, r, 'o', color=color, markersize=90, alpha=0.6)

    # --- Compute outward radial angle ---
    ang_left  = np.degrees(theta-diff)+180
    ang_right = np.degrees(theta+diff)

    ang_left_left  = np.degrees(theta-diff)+180
    ang_right_right= np.degrees(theta+diff)

    ang = np.degrees(theta)
    ang180 = np.degrees(theta)+180

    ax.scatter(theta-diff, r+0.1,
        marker=(8, 0, ang_left),   # triangle rotated by angle
        s=3500,
        c=[color],
        alpha=0.7)
    
    ax.scatter(theta+diff, r+0.1,
        marker=(8, 0, ang_right),   # triangle rotated by angle
        s=3500,
        c=[color],
        alpha=0.7)

    ax.scatter(theta-diff-diff, r+0.2,
        marker=(3, 0, ang_left_left),   # triangle rotated by angle
        s=3000,
        c=[color],
        alpha=0.7)
    
    ax.scatter(theta-diff+diff, r+0.2,
        marker=(3, 0, ang180),   # triangle rotated by angle
        s=3000,
        c=[color],
        alpha=0.7)

    ax.scatter(theta+diff+diff, r+0.2,
        marker=(3, 0, ang_right_right),   # triangle rotated by angle
        s=3000,
        c=[color],
        alpha=0.7)
    
    ax.scatter(theta+diff-diff, r+0.2,
        marker=(3, 0, ang),   # triangle rotated by angle
        s=3000,
        c=[color],
        alpha=0.7)
    
    # Labels
    ax.text(theta, r, str(oris[i]),
            fontsize=20, ha='center', va='center')
    ax.text((theta - diff), r+0.1, (str(oris[i]-8)), fontsize=15, ha='center', va='center', color='black', alpha=0.8)
    ax.text((theta + diff), r+0.1, (str(oris[i]+8)), fontsize=15, ha='center', va='center', color='black', alpha=0.8)
    ax.text((theta - diff-diff), r+0.2, ('false'), fontsize=15, ha='center', va='center', color='red', alpha=0.6)
    ax.text((theta + diff+diff), r+0.2, ('false'), fontsize=15, ha='center', va='center', color='red', alpha=0.9)
    ax.text((theta + diff-diff), r+0.2, ('correct'), fontsize=15, ha='center', va='center', color='green', alpha=0.9)


ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')

plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

colors = sns.diverging_palette(225, 15, s=70, n=4, l=50, center='light')

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')

ax.set_yticklabels([])
#ax.set_facecolor('None')
# ax.set_ylim(0, 0.7)

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.spines['polar'].set_visible(False)
#ax.grid(False)
#ax.set_rticks([])
#ax.xaxis.grid(True, color='pink', linestyle='--')

for i,j in enumerate(oris[2:4]):
    theta = np.radians(j)
    diff = np.radians(8)
    diff2 = np.radians(2)

    r = 0.3
    
    color = colors[i-1]

    # Main dot
    ax.plot(theta, r, 'o', color=color, markersize=60, alpha=1)

    # --- Compute outward radial angle ---
    ang_left  = np.degrees(theta-diff)+180
    ang_right = np.degrees(theta+diff)

    ang_left_left  = np.degrees(theta-diff-diff2)+180
    ang_right_right= np.degrees(theta+diff+diff2)
    ang_left_right  = np.degrees(theta-diff+diff2)
    ang_right_left= np.degrees(theta+diff-diff2)+180

    ang_s = np.degrees(theta-diff)-45
    ang_s2 = np.degrees(theta+diff)+45

    ang = np.degrees(theta)
    ang180 = np.degrees(theta)+180

    ax.scatter(theta-diff, r+0.13,
        marker=(4, 0, ang_s),   # triangle rotated by angle
        s=1800,
        c=[color],
        alpha=0.8)
    
    ax.scatter(theta+diff, r+0.13,
        marker=(4, 0, ang_s2),   # triangle rotated by angle
        s=1800,
        c=[color],
        alpha=0.8)

    ax.scatter(theta-diff-diff2, r+0.18,
        marker=(3, 0, ang_left_left),   # triangle rotated by angle
        s=70,
        c='red',
        alpha=1)
    
    ax.scatter(theta-diff+diff2, r+0.18,
        marker=(3, 0, ang_left_right),   # triangle rotated by angle
        s=70,
        c='green',
        alpha=1)

    ax.scatter(theta+diff+diff2, r+0.18,
        marker=(3, 0, ang_right_right),   # triangle rotated by angle
        s=70,
        c='red',
        alpha=1)
    
    ax.scatter(theta+diff-diff2, r+0.18,
        marker=(3, 0, ang_right_left),   # triangle rotated by angle
        s=70,
        c='green',
        alpha=1)
    
    tiny = np.radians(1.6)

    ax.plot([theta-diff+diff2-tiny, theta-diff+diff2], [r+0.18, r+0.18], linestyle='-', color='green', clip_on=False)
    ax.plot([theta-diff-diff2, theta-diff-diff2+ tiny], [r+0.18, r+0.18], linestyle='-', color='red', clip_on=False)

    ax.plot([theta+diff-diff2, theta+diff-diff2+tiny], [r+0.18, r+0.18], linestyle='-', color='green', clip_on=False)
    ax.plot([theta+diff+diff2-tiny, theta+diff+diff2], [r+0.18, r+0.18], linestyle='-', color='red', clip_on=False)
    
    # Labels
    ax.text(theta, r, str(j), fontsize=15, ha='center', va='center')
    ax.text((theta - diff), r+0.13, (str('away')), fontsize=8, ha='center', va='center', color='black', alpha=1)
    ax.text((theta + diff), r+0.13, (str('towards')), fontsize=8, ha='center', va='center', color='black', alpha=1)


ax.set_xticks([np.pi/2, np.pi/4])
ax.tick_params(axis='x', colors='black', labelsize=13)

ax.set_thetamin(45)
ax.set_thetamax(90)

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

colors = plt.cm.jet(np.linspace(0, 1, 17))

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='polar')

#ax.set_yticklabels([])
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)

ax.spines['polar'].set_visible(False)
ax.grid(False)
ax.set_rticks([])
#ax.xaxis.grid(True, color='pink', linestyle='--')

for i, j in enumerate(oris[2:4]):
    theta = np.radians(j)
    diff = np.radians(8)

    r = 0.2
    
    color = colors[i-1]

    # Main dot
    ax.plot(theta, r, 'o', color=color, markersize=75, alpha=0.6)

    # --- Compute outward radial angle ---
    ang_left  = np.degrees(theta-diff)+180
    ang_right = np.degrees(theta+diff)

    ang_left_left  = np.degrees(theta-diff)+180
    ang_right_right= np.degrees(theta+diff)

    ang = np.degrees(theta)
    ang180 = np.degrees(theta)+180

    ax.scatter(theta-diff, r+0.12,
        marker=(8, 0, ang_left),   # triangle rotated by angle
        s=2900,
        c=[color],
        alpha=0.7)
    
    ax.scatter(theta+diff, r+0.12,
        marker=(8, 0, ang_right),   # triangle rotated by angle
        s=2900,
        c=[color],
        alpha=0.7)

    ax.scatter(theta-diff-diff, r+0.3,
        marker=(3, 0, ang_left_left),   # triangle rotated by angle
        s=2500,
        c=[color],
        alpha=0.7)
    
    ax.scatter(theta-diff+diff, r+0.3,
        marker=(3, 0, ang180),   # triangle rotated by angle
        s=2500,
        c=[color],
        alpha=0.7)

    ax.scatter(theta+diff+diff, r+0.3,
        marker=(3, 0, ang_right_right),   # triangle rotated by angle
        s=2500,
        c=[color],
        alpha=0.7)
    
    ax.scatter(theta+diff-diff, r+0.3,
        marker=(3, 0, ang),   # triangle rotated by angle
        s=2500,
        c=[color],
        alpha=0.7)
    
    
    
    # Labels
    ax.text(theta, r, str(j),
            fontsize=20, ha='center', va='center')
    ax.text((theta - diff), r+0.12, (str(j-8)), fontsize=10, ha='center', va='center', color='black', alpha=1)
    ax.text((theta + diff), r+0.12, (str(j+8)), fontsize=10, ha='center', va='center', color='black', alpha=1)


ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

colors1 = sns.diverging_palette(225, 15, s=70, n=2, l=50, center='light')
colors2 = sns.diverging_palette(225, 15, s=70, n=2, l=50, center='light')
colors2.reverse()

colors = np.vstack((colors1, colors2)*10)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='polar')

ax.set_yticklabels([])
#ax.set_facecolor('None')
# ax.set_ylim(0, 0.7)

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.spines['polar'].set_visible(False)
#ax.grid(False)
#ax.set_rticks([])
#ax.xaxis.grid(True, color='pink', linestyle='--')

for i,j in enumerate(oris):
    theta = np.radians(j)
    diff = np.radians(8)
    diff2 = np.radians(2)

    r = 0.3
    
    color = colors[i]

    # Main dot
    ax.plot(theta, r, 'o', color=color, markersize=60, alpha=1)

    # --- Compute outward radial angle ---
    ang_left  = np.degrees(theta-diff)+180
    ang_right = np.degrees(theta+diff)

    ang_left_left  = np.degrees(theta-diff-diff2)+180
    ang_right_right= np.degrees(theta+diff+diff2)
    ang_left_right  = np.degrees(theta-diff+diff2)
    ang_right_left= np.degrees(theta+diff-diff2)+180

    ang_s = np.degrees(theta-diff)-45
    ang_s2 = np.degrees(theta+diff)+45

    ang = np.degrees(theta)
    ang180 = np.degrees(theta)+180

    ax.scatter(theta-diff, r+0.13,
        marker=(4, 0, ang_s),   # triangle rotated by angle
        s=1800,
        c=[color],
        alpha=0.8)
    
    ax.scatter(theta+diff, r+0.13,
        marker=(4, 0, ang_s2),   # triangle rotated by angle
        s=1800,
        c=[color],
        alpha=0.8)

    # ax.scatter(theta-diff-diff2, r+0.18,
    #     marker=(3, 0, ang_left_left),   # triangle rotated by angle
    #     s=70,
    #     c='red',
    #     alpha=1)
    
    # ax.scatter(theta-diff+diff2, r+0.18,
    #     marker=(3, 0, ang_left_right),   # triangle rotated by angle
    #     s=70,
    #     c='green',
    #     alpha=1)

    # ax.scatter(theta+diff+diff2, r+0.18,
    #     marker=(3, 0, ang_right_right),   # triangle rotated by angle
    #     s=70,
    #     c='red',
    #     alpha=1)
    
    # ax.scatter(theta+diff-diff2, r+0.18,
    #     marker=(3, 0, ang_right_left),   # triangle rotated by angle
    #     s=70,
    #     c='green',
    #     alpha=1)
    
    # tiny = np.radians(1.6)

    # ax.plot([theta-diff+diff2-tiny, theta-diff+diff2], [r+0.18, r+0.18], linestyle='-', color='green', clip_on=False)
    # ax.plot([theta-diff-diff2, theta-diff-diff2+ tiny], [r+0.18, r+0.18], linestyle='-', color='red', clip_on=False)

    # ax.plot([theta+diff-diff2, theta+diff-diff2+tiny], [r+0.18, r+0.18], linestyle='-', color='green', clip_on=False)
    # ax.plot([theta+diff+diff2-tiny, theta+diff+diff2], [r+0.18, r+0.18], linestyle='-', color='red', clip_on=False)
    
    # Labels
    ax.text(theta, r, str(j), fontsize=15, ha='center', va='center')
    ax.text((theta - diff), r+0.13, (str(j-8)), fontsize=10, ha='center', va='center', color='black', alpha=1)
    ax.text((theta + diff), r+0.13, (str(j+8)), fontsize=10, ha='center', va='center', color='black', alpha=1)


# ax.set_xticks([np.pi/2, np.pi/4])
# ax.tick_params(axis='x', colors='black', labelsize=13)

# # ax.set_thetamin(45)
# # ax.set_thetamax(90)
ax.set_rmax(0.5)
# ax.set_rmin(0)
# ax.set_rorigin(0)
# ax.grid(True)
# y_ticks = np.arange(0.3, 0.75, 0.1)
# ax.set_yticks(y_ticks)

# handles, labels = ax.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), loc='center')

plt.show()


# %%
# Set up the color wheel map
unbiased = np.full(17, 0.4)
colors = plt.cm.jet(np.linspace(0, 1, len(unbiased)))

fig = plt.figure()

oribins = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', ]

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
# Set the background color to None (transparent)
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)
# Plot the data with different colors
for i in range(len(oris)):
    theta = np.radians(oris[i])
    unbiased_assumption = np.full(17, 0.4)[i]
    ax.plot(theta, unbiased_assumption, 'o', color=colors[i], markersize=18)  # Use colors from the color wheel map

     # Add labels to the points inside the circle
    label_offset = 0.2  # Adjust this value for label position
    ax.text(theta, unbiased_assumption + label_offset, str(oribins[i]), fontsize=11, ha='center', va='center', color='black')

offset_outline_circle = np.full(360, 0.6)
spacing_outline_circle = np.arange(0, 360, 1)
spacing_outline_circle_rad = np.radians(spacing_outline_circle)

# plotting outline circle
ax.plot(spacing_outline_circle_rad, offset_outline_circle, 'k-')


# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
ax.xaxis.grid(color='pink', linestyle='-', linewidth=1)  #
# theta_circle = np.linspace(0, 2 * np.pi, 1000)
# r_circle = np.full(1000, np.pi)  # Adjust the radius as needed
# ax.plot(theta_circle, r_circle, 'b-', color='black')
# You can add a legend here if needed
# plt.legend(['Legend Name'])
plt.show()

# %%
# Set up the color wheel map
# old_colors = plt.cm.jet(np.linspace(0, 1, 17))
colors = sns.hls_palette(16)
oribins = ['Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal', ]

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
# Set the background color to None (transparent)
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)

# Remove the outer circle (frame) and grid lines
ax.spines['polar'].set_visible(False)  # Remove outer black circle
ax.grid(False)  # Remove grid lines
ax.set_rticks([])  # Remove radial ticks and grid lines

# Keep the cardinal (angular) grid lines
ax.xaxis.grid(True, color='pink', linestyle='--')  # Enable grid lines for angles

# Plot the data with different colors
for i in range(len(oris)):
    theta = np.radians(oris[i])
    l = np.full(17, 0.55)[i]
    ax.plot(theta, l, 'o', color=colors[i], linewidth=0.2, markersize=38, alpha=0.6)  # Use colors from the color wheel map
    
    # Add labels to the points inside the circle
    label_offset = 0.0  # Adjust this value for label position
    ax.text(theta, l + label_offset, str(oris[i]), fontsize=9, ha='center', va='center', color='black')

     # Add labels to the points inside the circle
    label_offset = 0.21  # Adjust this value for label position
    ax.text(theta, l + label_offset, str(oribins[i]), fontsize=10, ha='center', va='center', color='grey')

# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
plt.savefig('../figures/colorful_dots_cardinal_bins.png')
plt.show()

# %%
# Set up the color wheel map
colors = plt.cm.jet(np.linspace(0, 1, 17))
oribins = ['Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal', ]

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
# Set the background color to None (transparent)
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)

# Remove the outer circle (frame) and grid lines
ax.spines['polar'].set_visible(False)  # Remove outer black circle
ax.grid(False)  # Remove grid lines
ax.set_rticks([])  # Remove radial ticks and grid lines

# Keep the cardinal (angular) grid lines
ax.xaxis.grid(True, color='pink', linestyle='--')  # Enable grid lines for angles

# Plot the data with different colors
for i in range(len(oris)):
    theta = np.radians(oris[i])
    l = np.full(17, 0.55)[i]
    ax.plot(theta, l, 'o', color=colors[i], linewidth=0.2, markersize=8, alpha=0.6)  # Use colors from the color wheel map
     # Add labels to the points inside the circle
    label_offset = 0.0  # Adjust this value for label position
    ax.text(theta, l + label_offset, str(oris[i]), fontsize=9, ha='center', va='center', color='black')

     # Add labels to the points inside the circle
    label_offset = 0.21  # Adjust this value for label position
    ax.text(theta, l + label_offset, str(oribins[i]), fontsize=10, ha='center', va='center', color='grey')

# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
plt.savefig('../figures/colorful_dots_cardinal_bins.png')
plt.show()

# %%
from scipy.stats import vonmises
import seaborn as sns



df = pd.read_csv('../data/csv_files/df.csv')
partis = df['subID'].unique()

# Get ordered list of unique target orientations
oris = df['object_cued_rot'].unique()
oris.sort()   

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')

# Get outer ylim for plotting radial lines
ylim = ax.get_ylim()

# Plot a radial line for each target orientation - colour matched to its kde distribution
# Then plot a radial line for the mean of each kde distribution
for i in range(len(oris)):
    theta = np.radians(oris[i])
    mean_bp = (df[df['object_cued_rot']==oris[i]]['button_press_clockwise']).mean()
   
    # ax.plot([theta, theta], [0, mean_bp], 'o--', c=sns.hls_palette(16)[r], alpha = 0.5, markersize=10)
    ax.bar([theta, theta], [0, mean_bp], color=sns.hls_palette(16)[i], alpha = 0.5, width = 0.3)

# Move r origin outwards to have a hollow circle in the middle for ~AesTHeTicS~
rorigin = -0.5*ylim[1]
ax.set_rorigin(rorigin)

# Remove yticks
ax.set_yticklabels([])
ax.set_yticks([])

# Set the direction of the zero angle
ax.set_theta_offset(np.pi/2)

plt.show()

# %%
from scipy.stats import vonmises
import seaborn as sns



df = pd.read_csv('../data/csv_files/df.csv')
partis = df['subID'].unique()

# Get ordered list of unique target orientations
oris = df['object_cued_rot'].unique()
oris.sort()   

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')

# Get outer ylim for plotting radial lines
ylim = ax.get_ylim()

# Plot a radial line for each target orientation - colour matched to its kde distribution
# Then plot a radial line for the mean of each kde distribution
for i in range(len(oris)):
    theta = np.radians(oris[i])
    mean_bp = (df[df['object_cued_rot']==oris[i]]['button_press_clockwise']).mean()
   
    ax.plot([theta, theta], [0, mean_bp], 'o-', c=sns.hls_palette(16)[i], alpha = 0.5, markersize=10)

# Move r origin outwards to have a hollow circle in the middle for ~AesTHeTicS~
rorigin = -0.5*ylim[1]
ax.set_rorigin(rorigin)

# Remove yticks
ax.set_yticklabels([])
ax.set_yticks([])

# Set the direction of the zero angle
ax.set_theta_offset(np.pi/2)

plt.show()

# %%
# # Get ordered list of unique target orientations
# oris = Juan_df['object_cued_rot'].unique()
# oris.sort()
# oris = np.append(oris, oris[0])

# fig = plt.figure()

# # Change the projection to 'polar'
# ax = fig.add_subplot(111, projection='polar')

# # Get outer ylim for plotting radial lines
# ylim = ax.get_ylim()

# # Arrays to store the angles and mean button presses
# thetas = []
# mean_bps = []

# # Plot a radial line for each target orientation - colour matched to its kde distribution
# # Then plot a radial line for the mean of each kde distribution
# for i in range(len(oris)):
#     theta = np.radians(oris[i])
#     mean_bp = (Juan_df[Juan_df['object_cued_rot']==oris[i]]['button_press_clockwise']).mean()
#     thetas.append(theta)
#     mean_bps.append(mean_bp)

#     ax.plot(theta, mean_bp, 'go')

#     text_positions = np.full(17, 0.55)[i]
#     label_offset = 0.0  # Adjust this value for label position
#     ax.text(theta, text_positions + label_offset, str(oris[i]), fontsize=9, ha='center', va='center', color='black')

# ax.plot(thetas, mean_bps, 'g-', linewidth=2)

# unbiased_assumption = np.full(17, 0.5)


from scipy.stats import vonmises
import seaborn as sns



df = pd.read_csv('../data/csv_files/df.csv')
partis = df['subID'].unique()

# Get ordered list of unique target orientations
oris = df['object_cued_rot'].unique()
oris.sort()
oris = np.append(oris, oris[0])

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')

# Get outer ylim for plotting radial lines
ylim = ax.get_ylim()

thetas = []
mean_bps = []
for i in range(len(oris)):
    theta = np.radians(oris[i])
    mean_bp = (df[df['object_cued_rot']==oris[i]]['button_press_clockwise']).mean()
    thetas.append(theta)
    mean_bps.append(mean_bp)

    ax.plot(theta, mean_bp, 'go')

    text_positions = np.full(17, 0.55)[i]
    label_offset = 0.0  # Adjust this value for label position
    ax.text(theta, text_positions + label_offset, str(oris[i]), fontsize=9, ha='center', va='center', color='black')

ax.plot(thetas, mean_bps, 'g-', linewidth=2)

unbiased_assumption = np.full(17, 0.5)
ax.plot(thetas, unbiased_assumption, 'k--')

# # Move r origin outwards to have a hollow circle in the middle for ~AesTHeTicS~
# rorigin = -0.5*ylim[1]
# ax.set_rorigin(rorigin)

# # Remove yticks
# ax.set_yticklabels([])
# ax.set_yticks([])

# Set the direction of the zero angle
ax.set_theta_offset(np.pi/2)

plt.show()

# %%
# Set up the color wheel map
# old_colors = plt.cm.jet(np.linspace(0, 1, 17))
colors = sns.hls_palette(16)
oribins = ['Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal','Cardinal', 'Oblique', 'Oblique', 'Cardinal', ]

ori_diffs = np.array(oris) +11.25

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')
ax.set_yticklabels([])
# Set the background color to None (transparent)
ax.set_facecolor('None')
ax.set_ylim(0, 0.7)

# Remove the outer circle (frame) and grid lines
ax.spines['polar'].set_visible(False)  # Remove outer black circle
ax.grid(False)  # Remove grid lines
ax.set_rticks([])  # Remove radial ticks and grid lines

# Keep the cardinal (angular) grid lines
ax.xaxis.grid(True, color='pink', linestyle='--')  # Enable grid lines for angles

# Plot the data with different colors
for i in range(len(oris)):
    theta = np.radians(oris[i])
    theta_diffs = np.radians(ori_diffs[i])
    l = np.full(17, 0.55)[i]
    ax.plot(theta, l, 'o', color=colors[i], linewidth=0.2, markersize=38, alpha=0.6)
    ax.plot(theta_diffs, i, 'o', color=colors[i], linewidth=0.2, markersize=38, alpha=0.6)
    # Add labels to the points inside the circle
    label_offset = 0.0  # Adjust this value for label position
    ax.text(theta, l + label_offset, str(oris[i]), fontsize=8, ha='center', va='center', color='grey')
    ax.text(theta_diffs, l + label_offset, str(ori_diffs[i]), fontsize=8, ha='center', va='center', color='black')

     # Add labels to the points inside the circle
    label_offset = 0.21  # Adjust this value for label position
    ax.text(theta, l + label_offset, str(oribins[i]), fontsize=10, ha='center', va='center', color='grey')

# Adding lines and labels for the cardinal axes
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.tick_params(axis='x', colors='pink')  # Color of the tick labels
plt.savefig('../figures/colorful_dots_cardinal_bins.png')
plt.show()

# %%
# %%
# Plot where the original orientations of the data are plotted to inspect


# Get ordered list of unique target orientations
oris = df['object_cued_rot'].unique()
oris.sort()
oris = np.append(oris, oris[0])

fig = plt.figure()

# Change the projection to 'polar'
ax = fig.add_subplot(111, projection='polar')

# Get outer ylim for plotting radial lines
ylim = ax.get_ylim()

# Arrays to store the angles and mean button presses
thetas = []
thetas_adjusted = []
mean_bps = []

# Plot the first point with a different color (e.g., red)
theta_first = np.radians(oris[0])
# Adjust theta for actual data to be plotted starting at an offset of 90° (+11.25) in anticlockwise direction
theta_first_adjusted = -theta_first + ((np.pi / 2)) % (2 * np.pi)
mean_bp_first = df[df['object_cued_rot'] == oris[0]]['button_press_clockwise'].mean()
ax.plot(theta_first_adjusted, (mean_bp_first), 'ro', markersize=14)

# Store the first values
thetas.append(theta_first)
thetas_adjusted.append(theta_first_adjusted)
mean_bps.append(mean_bp_first)

# Plot the remaining points and labels
for i in range(1, len(oris)):
    theta = np.radians(oris[i])
    theta_adjusted = -theta + ((np.pi/2)) % (2*np.pi)

    # Optional to print out the adjusted theta in degees
    # print(np.degrees(theta))
    # theta_adjusted = (theta +(np.pi/2))
    # print(np.degrees(theta_adjusted))
    # theta_adjusted = theta_adjusted % (2 * np.pi)
    # print(np.degrees(theta_adjusted))
    # theta_adjusted = -theta_adjusted
    # print(np.degrees(theta_adjusted))
    # print('______')

    mean_bp = df[df['object_cued_rot'] == oris[i]]['button_press_clockwise'].mean()
    thetas.append(theta)
    thetas_adjusted.append(theta_adjusted)
    mean_bps.append((mean_bp))

    # Plot the remaining points in blue
    ax.plot(theta_adjusted, (mean_bp), 'go')

    # Add text labels for each point
    text_positions = np.full(17, 0.55)[i]
    label_offset = 0.0  # Adjust this value for label position
    ax.text(theta_adjusted, text_positions + label_offset, str(oris[i]), fontsize=9, ha='center', va='center', color='black')

# Plot the line connecting all points
ax.plot(thetas_adjusted, mean_bps, 'g-', linewidth=2)
ax.plot(thetas_adjusted, unbiased_assumption, 'k--')

# Saving the plot
# plt.savefig(os.path.join(figures_save_dir, 'manual_theta_manipulation_with_ori_labels_plotted.png'))

plt.show()


