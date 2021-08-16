#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import copy

import cv2
import numpy
import scipy.stats

import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.api import add_constant, OLS
from statsmodels.stats.outliers_influence import summary_table
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from umap import UMAP

from reliability import test_retest
from MouseViewParser.readers import gorilla
from pygazeanalyser.opengazereader import read_opengaze

pyplot.rc('text', usetex=True)


# # # # #
# CONSTANTS

# BASIC CONTROLS

# EXPERIMENT SETTINGS
# Specify the name of the zone that starts the first trial.
TRIAL_START_ZONE = "Zone3"
# Specify the custom fields that should be loaded from the data.
CUSTOM_FIELDS = ["left_stim", "right_stim", "condition", "side"]
# Stimuli for each condition.
STIMULI = { \
    "disgust":  ["disgust1", "disgust2", "disgust3", "disgust4", "disgust5"], \
    "pleasant": ["pleasant1", "pleasant2", "pleasant3", "pleasant4", \
        "pleasant5"], \
    }
# Conditions.
CONDITIONS = list(STIMULI.keys())
CONDITIONS.sort()
# Areas of interest.
AOI = ["affective", "neutral", "other"]
# Number of trials per condition.
NTRIALS = 20
# Number of trials per stimulus (affective stimuli get repeated).
NTRIALSPERSTIM = 4
# Trial duration in milliseconds.
TRIAL_DURATION = 10000

# ANALYSIS SETTINGS
# Bin size for re-referencing samples to bins across the trial duration.
# This is in milliseconds.
BINWIDTH = 100.0 / 3.0
# Multiple-comparisons correction. Choose from None (alpha=0.05), "bonferroni",
# "holm", or a float between 0 and 1 to set the alpha level directly.
MULTIPLE_COMPARISONS_CORRECTION = "holm"
# Set the regression backend to use. It can be set to "statsmodels", and
# anything else will result in the default SciPy implementation.
REGRESSION_BACKEND = "statsmodels"
# RESCALE DISPLAY SIZE
VIEWPORT = (1920, 937)
VIEWBINS = (VIEWPORT[0]//30, VIEWPORT[1]//30)
STIMRECT = { \
    "affective": (401,  289, 480, 360), \
    "neutral":   (1038, 289, 480, 360), \
    }

# FILES AND FOLDERS.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data")
RATINGSPATH = os.path.join(DATADIR, "ratings_{}_memmap.dat")
RATINGSSHAPEPATH = os.path.join(DATADIR, "ratings_shape_{}_memmap.dat")
DWELLPATH = os.path.join(DATADIR, "dwell_mouse_memmap.dat")
DWELLSHAPEPATH = os.path.join(DATADIR, "dwell_mouse_shape_memmap.dat")
GAZEDIR = os.path.join(DIR, "gaze_data")
GAZEPATH = os.path.join(DATADIR, "dwell_gaze_memmap.dat")
GAZESHAPEPATH = os.path.join(DATADIR, "dwell_gaze_shape_memmap.dat")
RESAMPLEPATH = os.path.join(DATADIR, "resampled_xy_{}.dat")
RESAMPLESHAPEPATH = os.path.join(DATADIR, "resampled_xy_shape_{}.dat")
MDSPATH = os.path.join(DATADIR, "MDS_reduced_xy.dat")
UMAPPATH = os.path.join(DATADIR, "UMAP_reduced_xy.dat")
OUTDIR = os.path.join(DIR, "output")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# PLOTTING
PLOTCOLMAPS = { \
    "disgust":  "Oranges", \
    "pleasant": "Blues", \
    "heatmap":  "BrBG", \
    }
# Font sizes for different elements.
FONTSIZE = { \
    "title":            32, \
    "axtitle":          24, \
    "legend":           14, \
    "bar":              14, \
    "label":            16, \
    "ticklabels":       12, \
    "annotation":       14, \
    }
# Font sizes if we're using LaTex.
FONTSIZE = { \
    "title":            32, \
    "axtitle":          28, \
    "legend":           18, \
    "bar":              20, \
    "label":            20, \
    "ticklabels":       14, \
    "annotation":       18, \
    }
# Set the y limits for various plots.
YLIM = { \
    "dwell_p":  [-50, 60], \
    "welch":    [-2, 2], \
    "bayes":    [-5, 5], \
    }
# Set the locations of legends.
LEGENDLOCS = { \
    "disgust":  "lower right", \
    "pleasant": "lower right", \
    }


# # # # #
# LOAD DATA

# RATINGS
ratings = {}

# Load rating data.
for data_type in ["gaze", "mouse"]:
    ratings_shape = tuple(numpy.memmap(RATINGSSHAPEPATH.format(data_type), \
        dtype=numpy.int16, mode="r"))
    ratings[data_type] = numpy.memmap(RATINGSPATH.format(data_type), \
        dtype=numpy.float32, mode="r", shape=ratings_shape)

# DWELL TIMES
dwell = {}

# Load the dwell data's shape from the dwell_shape file.
gaze_shape = tuple(numpy.memmap(GAZESHAPEPATH, dtype=numpy.int32, mode="r"))
# Load the dwell data from file.
dwell["gaze"] = numpy.memmap(GAZEPATH, dtype=numpy.float32, mode="r", \
    shape=gaze_shape)

# Load the dwell data's shape from the dwell_shape file.
mouse_shape = tuple(numpy.memmap(DWELLSHAPEPATH, dtype=numpy.int32, mode="r"))
# Load the dwell data from file.
dwell["mouse"] = numpy.memmap(DWELLPATH, dtype=numpy.float32, mode="r", \
    shape=mouse_shape)

# Recompute the bin edges.
bin_edges = numpy.arange(0, TRIAL_DURATION+BINWIDTH/2, BINWIDTH, \
    dtype=numpy.float32)

# SCAN PATHS
scan = {}
# Load scan paths.
for data_type in ["mouse", "gaze"]:
    # The shape is (n_participants, x=0/y=1, n_conditions, n_stimuli,
    # n_presentations, n_samples)
    scan_shape = tuple(numpy.memmap(RESAMPLESHAPEPATH.format(data_type), \
        dtype=numpy.int32, mode="r"))
    scan[data_type] = numpy.memmap(RESAMPLEPATH.format(data_type), \
        dtype=numpy.float32, mode="r", shape=scan_shape)


# # # # #
# RATING CORRELATIONS

# Compute gaze differences.
d = {}
for key in dwell.keys():
    d[key] = numpy.nanmean(dwell[key][:,:,:,:,0,:] - dwell[key][:,:,:,:,1,:], \
        axis=4)
# Compute correlations between stimulus and dwell time difference for all
# stimuli, separated for each presentation.
n_participants, n_conditions, n_stimuli, n_repetitions = \
    dwell["gaze"].shape[0:4]
with open(os.path.join(OUTDIR, "rating_dwell_correlations.csv"), "w") as f:
    # Write header to file.
    header = ["condition", "stimulus", "presentation_nr", "r_gaze", "p", \
        "r_mouse", "p", "z_difference", "p", "tau_gaze", "p", "tau_mouse", "p"]
    f.write(",".join(header))
    # Loop through all conditions.
    for ci, condition in enumerate(CONDITIONS):
        # Loop through all stimuli.
        for si, stimulus in enumerate(STIMULI[condition]):
            
            # Loop through all repetitions.
            for ri in range(n_repetitions+1):
                
                # Select the data for this repetition, or average across all
                # repetitions if this is the final iteration.
                if ri == n_repetitions:
                    # Grab ratings, and compute dwell difference averages.
                    x_gaze = ratings["gaze"][:,ci,si]
                    y_gaze = numpy.nanmean(d["gaze"][:,ci,si,:], axis=1)
                    x_mouse = ratings["mouse"][:,ci,si]
                    y_mouse = numpy.nanmean(d["mouse"][:,ci,si,:], axis=1)
                    # Filter out NaNs.
                    notnan_gaze = (numpy.isnan(x_gaze)==False) & \
                        (numpy.isnan(y_gaze)==False)
                    x_gaze = x_gaze[notnan_gaze]
                    y_gaze = y_gaze[notnan_gaze]
                    notnan_mouse = (numpy.isnan(x_mouse)==False) & \
                        (numpy.isnan(y_mouse)==False)
                    x_mouse = x_mouse[notnan_mouse]
                    y_mouse = y_mouse[notnan_mouse]
                else:
                    # Find NaNs.
                    notnan_gaze = \
                        (numpy.isnan(ratings["gaze"][:,ci,si])==False) \
                        & (numpy.isnan(d["gaze"][:,ci,si,ri])==False)
                    notnan_mouse = \
                        (numpy.isnan(ratings["mouse"][:,ci,si])==False) \
                        & (numpy.isnan(d["mouse"][:,ci,si,ri])==False)
                    # Grab the data, and filter out NaNs.
                    x_gaze = ratings["gaze"][:,ci,si][notnan_gaze]
                    y_gaze = d["gaze"][:,ci,si,ri][notnan_gaze]
                    x_mouse = ratings["mouse"][:,ci,si][notnan_mouse]
                    y_mouse = d["mouse"][:,ci,si,ri][notnan_mouse]

                # Compute Pearson R for both conditions.
                r_gaze, p_gaze = scipy.stats.pearsonr(x_gaze, y_gaze)
                r_mouse, p_mouse = scipy.stats.pearsonr(x_mouse, y_mouse)

                # Compute difference between correlations using Fisher
                # z-transformation.
                z_gaze = r_gaze = 0.5 * numpy.log((1+r_gaze)/(1-r_gaze))
                n_gaze = numpy.sum(notnan_gaze.astype(int))
                z_mouse = r_mouse = 0.5 * numpy.log((1+r_mouse)/(1-r_mouse))
                n_mouse = numpy.sum(notnan_mouse.astype(int))
                sd = numpy.sqrt((1.0 / (n_gaze-3)) + (1.0 / (n_mouse-3)))
                z = (z_gaze - z_mouse) / sd
                p = 2 * (1 - scipy.stats.norm.cdf(numpy.abs(z)))

                # Compute Kendall's tau for both.
                tau_gaze, p_tau_gaze = scipy.stats.kendalltau(x_gaze, y_gaze)
                tau_mouse, p_tau_mouse = scipy.stats.kendalltau(x_mouse, \
                    y_mouse)

                # Write line to file.
                if ri == n_repetitions:
                    rep = "AVG"
                else:
                    rep = ri+1
                line = [condition, stimulus, rep, r_gaze, p_gaze, r_mouse, \
                    p_mouse, z, p, tau_gaze, p_tau_gaze, tau_mouse, \
                    p_tau_mouse]
                f.write("\n" + ",".join(map(str, line)))

# PLOT RATING * DWELL
fig, ax = pyplot.subplots(nrows=len(CONDITIONS), ncols=2, \
    figsize=(16.0,6.0*len(CONDITIONS)), dpi=100.0)
fig.subplots_adjust(left=0.07, bottom=0.07, right=0.95, top=0.95,
    wspace=0.15, hspace=0.2)
# Run through all conditions.
for ci, condition in enumerate(CONDITIONS):
    
    # Specify the colour map for the current condition.
    cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
    voffset = 3
    vmin = 0
    vmax = dwell["gaze"].shape[3] + voffset
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Determine the colour for this panel.
    col = cmap(norm(voffset))
    
    # Loop through both data types.
    r = {}
    n = {}
    for di, data_type in enumerate(["gaze", "mouse"]):

        # Compute the average rating and dwell time differences. Dwell 
        # difference is multiplied by 100 to make it reflect percentage points.
        x_vals = numpy.nanmean(ratings[data_type][:,ci,:], axis=1)
        y_vals = numpy.nanmean(numpy.nanmean(d[data_type][:,ci,:,:]*100, \
            axis=1), axis=1)
    
        # Find the order of all x-values, for plotting purposes.
        # NOTE: x will be used to order, and y values will be ordered
        # according to x's order. Hence, the correct values will remain
        # linked. Separate ordering would be exceptionally wrong.
        notnan = (numpy.isnan(x_vals)==False) & (numpy.isnan(y_vals)==False)
        n[data_type] = numpy.sum(notnan.astype(int))
        order = numpy.argsort(x_vals[notnan])
        x = add_constant(x_vals[notnan][order])
        y = y_vals[notnan][order]
        
        # Compute correlation coefficients.
        r[data_type], p = scipy.stats.pearsonr(x[:,1], y)

        # Construct the correlation value strings.
        if r[data_type] < 0:
            r_str = "$R=${}".format(str(round(r[data_type], 2)).ljust(5, '0'))
        else:
            r_str = "$R=${}".format(str(round(r[data_type], 2)).ljust(4, '0'))
        if p < 0.001:
            p_str = r"$p<$0.001"
        else:
            p_str = "$p=${}".format(str(round(p, 3)).ljust(5, '0'))
        # Construct the label for the regression line.
        if di == 1:
            # Compute difference between correlations using Fisher
            # z-transformation.
            z_gaze = 0.5 * numpy.log((1+r["gaze"])/(1-r["gaze"]))
            z_mouse = 0.5 * numpy.log((1+r["mouse"])/(1-r["mouse"]))
            sd = numpy.sqrt((1.0 / (n["gaze"]-3)) + (1.0 / (n["mouse"]-3)))
            z = (z_gaze - z_mouse) / sd
            zp = 2 * (1 - scipy.stats.norm.cdf(numpy.abs(z)))
            if z < 0:
                z_str = r"$Z_{gaze-mouse}=$" + \
                    "{}".format(str(round(z, 2)).ljust(5, '0'))
            else:
                z_str = r"$Z_{gaze-mouse}$" + \
                    "={}".format(str(round(z, 2)).ljust(4, '0'))
            if zp < 0.001:
                zp_str = "$p<$0.001"
            else:
                zp_str = "$p=${}".format(str(round(zp, 3)).ljust(5, '0'))
            lbl = r"{}, {}; {}, {}".format(r_str, p_str, z_str, zp_str)
        else:
            lbl = r"{}, {}".format(r_str, p_str)

        # Plot the individual samples.
        ax[ci,di].plot(x[:,1], y, "o", color=col, alpha=0.3)
        
        # Run the regression.
        model = OLS(y, x)
        result = model.fit()
        # Get the fit details.
        st, dat, ss2 = summary_table(result, alpha=0.05)
        y_pred = dat[:,2]
        predict_mean_se  = dat[:,3]
        predict_mean_ci_low, predict_mean_ci_upp = dat[:,4:6].T
        predict_ci_low, predict_ci_upp = dat[:,6:8].T

        # Plot the regression line.
        ax[ci,di].plot(x[:,1], y_pred, "-", lw=3, color=col, alpha=0.8, \
            label=lbl)
        # Plot regression confidence envelope.
        ax[ci,di].fill_between(x[:,1], predict_mean_ci_low, \
            predict_mean_ci_upp, color=col, alpha=0.2)
        # Plot prediction intervals.
        ax[ci,di].plot(x[:,1], predict_ci_low, '-', lw=2, color=col, alpha=0.5)
        ax[ci,di].plot(x[:,1], predict_ci_upp, '-', lw=2, color=col, alpha=0.5)
        
        # Set plot title.
        if ci == 0:
            ax[ci,di].set_title(data_type.capitalize(), \
                fontsize=FONTSIZE["axtitle"])
        # Set plot x label.
        ax[ci,di].set_xlabel("Stimulus {} rating".format(condition), \
            fontsize=FONTSIZE["label"])
        
        # Set limits.
        if condition == "disgust":
            ax[ci,di].set_xlim([0,100])
        elif condition == "pleasant":
            ax[ci,di].set_xlim([-20,100])
        ax[ci,di].set_ylim([-100,100])
        # Set labels.
        ax[ci,di].set_xticklabels(ax[ci,di].get_xticks(), \
            fontsize=FONTSIZE["ticklabels"])
        ax[ci,di].set_yticklabels(ax[ci,di].get_yticks(), \
            fontsize=FONTSIZE["ticklabels"])
        
        # Add a legend.
        ax[ci,di].legend(loc="upper right", fontsize=FONTSIZE["legend"])
        
    # Set the y label.
    ax[ci,0].set_ylabel(r"$\Delta$dwell time ({}-neutral \% pt.)".format( \
        condition), fontsize=FONTSIZE["label"])

fig.savefig(os.path.join(OUTDIR, "ratings_dwell_correlation_comparison.png"))
pyplot.close(fig)


# # # # #
# SCANPATH PLOTS

# HEATMAPS
# Make heatmaps out of the scan paths.
heatmap = {}
for data_type in scan.keys():
    heatmap[data_type] = {}
    for ci in range(scan[data_type].shape[2]):
        condition = CONDITIONS[ci]
        shape = (VIEWBINS[1], VIEWBINS[0], scan[data_type].shape[3], \
            scan[data_type].shape[4])
        heatmap[data_type][condition] = numpy.zeros(shape, dtype=numpy.int16)
        for si in range(scan[data_type].shape[3]):
            for pi in range(scan[data_type].shape[4]):

                # Select all the values that are not NaN, and within the
                # display.
                not_nan = numpy.isnan(scan[data_type][:,:,ci,si,pi,:]) == False
                within_disp = (scan[data_type][:,:,ci,si,pi,:] > 0) & \
                    (scan[data_type][:,:,ci,si,pi,:] <= 1)
                sel = not_nan & within_disp
                sel = sel[:,0,:] & sel[:,1,:]
                
                # Create heatmap.
                heatmap[data_type][condition][:,:,si,pi], _, _ = \
                    numpy.histogram2d( \
                    scan[data_type][:,1,ci,si,pi,:][sel], \
                    scan[data_type][:,0,ci,si,pi,:][sel], \
                    range=[[0,1], [0,1]], \
                    bins=(VIEWBINS[1],VIEWBINS[0]), density=True)

# Plot heatmaps.
for ci, condition in enumerate(CONDITIONS):
    # One separate figure for each condition. Each figure will have two rows:
    # one for gaze and one for mouse heatmaps. There will be as many columns
    # as there are stimuli.
    fig, ax = pyplot.subplots(nrows=2, ncols=len(STIMULI[condition]), \
        figsize=(len(STIMULI[condition])*7.0, 12.0), dpi=100)
    fig.subplots_adjust(left=0.05, bottom=0.02, right=0.98, top=0.95,
        wspace=0.1, hspace=0.1)

    # Loop through all heatmaps.
    for mi, method in enumerate(["gaze", "mouse"]):
        # Loop through all stimuli.
        for si, stim_name in enumerate(STIMULI[condition]):
            
            # Load the stimulus image.
            img = matplotlib.pyplot.imread(os.path.join(DIR, "images", \
                "{}_{}.png".format(method, stim_name)))
            # Blur the image (values are kernel size and SD).
            img = cv2.GaussianBlur(img, (51,51), 150.0)
            # Compute and rescale the heatmap.
            hm = numpy.nanmean(heatmap[method][condition][:,:,si,:], axis=2)
            hm = cv2.resize(hm, (img.shape[1],img.shape[0]), \
                interpolation=cv2.INTER_NEAREST)
            # Plot the heatmap and get the data.
            cmap = matplotlib.cm.get_cmap("viridis")
            norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
            hm = cmap(norm(hm))[:,:,:3]
            
            # Combine into a single matrix.
            m = img*0.4 + hm*0.6

            # Plot the image+heatmap.
            ax[mi,si].imshow(m)
            
            # Set the axis titles.
            if mi == 0:
                ax[mi,si].set_title(stim_name, fontsize=FONTSIZE["title"])
            if si == 0:
                ax[mi,si].set_ylabel(method.capitalize(), \
                    fontsize=FONTSIZE["title"])

            # Remove the axis ticks.
            ax[mi,si].set_xticks([])
            ax[mi,si].set_yticks([])
                
    # Save and close the figure.
    fig.savefig(os.path.join(OUTDIR, "heatmaps_{}.png".format(condition)))
    pyplot.close(fig)

# SCANPATH DIM REDUCTION
# Collect all scanpaths into a big matrix with participants in rows, and
# (x,y) and time as features in columns.
n_gaze = scan["gaze"].shape[0]
n_mouse = scan["mouse"].shape[0]
n_participants = n_gaze + n_mouse
n_conditions = scan["gaze"].shape[2]
n_stimuli = scan["gaze"].shape[3]
n_repeats = scan["gaze"].shape[4]
n_bins = scan["gaze"].shape[5]
shape = (n_participants*n_conditions*n_stimuli*n_repeats, 2*n_bins)
scanpath = { \
    "datatype":     numpy.zeros(shape[0], dtype=numpy.int16), \
    "condition":    numpy.zeros(shape[0], dtype=numpy.int16), \
    "stimulus":     numpy.zeros(shape[0], dtype=numpy.int16), \
    "repeat":       numpy.zeros(shape[0], dtype=numpy.int16), \
    "X":            numpy.zeros(shape, dtype=numpy.float64)
    }
start = 0
for ci in range(n_conditions):
    condition = CONDITIONS[ci]
    for si in range(n_stimuli):
        for ri in range(n_repeats):
            # Update the ending point.
            end = start + n_participants
            # Update the specifics.
            scanpath["datatype"][start:start+n_gaze] = 0
            scanpath["datatype"][start+n_gaze:start+n_gaze+n_mouse] = 1
            scanpath["condition"][start:end] = ci
            scanpath["stimulus"][start:end] = si
            scanpath["repeat"][start:end] = ri
            # Add the (x,y) coordinates across all time bins.
            scanpath["X"][start:start+n_gaze, :300] = \
                scan["gaze"][:,0,ci,si,ri,:]
            scanpath["X"][start:start+n_gaze, 300:] = \
                scan["gaze"][:,1,ci,si,ri,:]
            scanpath["X"][start+n_gaze:start+n_gaze+n_mouse, :300] = \
                scan["mouse"][:,0,ci,si,ri,:]
            scanpath["X"][start+n_gaze:start+n_gaze+n_mouse, 300:] = \
                scan["mouse"][:,1,ci,si,ri,:]
            # Update the starting point for the next iteration.
            start = copy.deepcopy(end)

# Filter out all NaN values.
include = numpy.sum(numpy.isnan(scanpath["X"]).astype(int), axis=1) == 0
for key in scanpath.keys():
    if key == "X":
        scanpath[key] = scanpath[key][include,:]
    else:
        scanpath[key] = scanpath[key][include]
n_included = int(numpy.sum(include.astype(int)))

# Apply multi-dimensional scaling, or load existing data.
if not os.path.isfile(MDSPATH):
    print("Projecting into 2 dimensions using MDS (this takes for ever)...")
    red = MDS(n_components=2)
    scanpath["X_reduced_mds"] = red.fit_transform(scanpath["X"])
    mds_mapped = numpy.memmap(MDSPATH, dtype=numpy.float64, mode="w+", \
        shape=(n_included,2))
    mds_mapped[:] = scanpath["X_reduced_mds"][:]
    mds_mapped.flush()
    print("Projecting into 2 dimensions using UMAP (this takes a while)...")
    red = UMAP(n_components=2)
    scanpath["X_reduced_umap"] = red.fit_transform(scanpath["X"])
    umap_mapped = numpy.memmap(UMAPPATH, dtype=numpy.float64, mode="w+", \
        shape=(n_included,2))
    umap_mapped[:] = scanpath["X_reduced_umap"][:]
    umap_mapped.flush()
else:
    mds_mapped = numpy.memmap(MDSPATH, dtype=numpy.float64, mode="r", \
        shape=(n_included,2))
    scanpath["X_reduced_mds"] = numpy.copy(mds_mapped)
    umap_mapped = numpy.memmap(UMAPPATH, dtype=numpy.float64, mode="r", \
        shape=(n_included,2))
    scanpath["X_reduced_umap"] = numpy.copy(umap_mapped)

# Plot the reduced spaces.
for method in ["mds", "umap"]:
    marker = ["o", "^", "p", "P", "*"]
    # Create a new figure.
    fig, ax = pyplot.subplots(nrows=2, ncols=2, \
        figsize=(16.0,12.0))
    fig.subplots_adjust(left=0.07, bottom=0.05, right=0.98, top=0.95,
        wspace=0.1, hspace=0.1)
    for ci in range(n_conditions):
        # Select the colourmap for this condition.
        condition = CONDITIONS[ci]
        cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
        voffset = 1
        vmin = 0
        vmax = n_repeats + voffset
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
        for di, datatype in enumerate(["gaze","mouse"]):
            # Plot all samples, but light and in grey.
            sel = scanpath["datatype"] != di
            ax[ci,di].plot( \
                scanpath["X_reduced_{}".format(method)][sel,0], \
                scanpath["X_reduced_{}".format(method)][sel,1], \
                ".", markersize=3, color="#AAAAAA", alpha=0.1)
            for si in range(n_stimuli):
                for ri in range(n_repeats):
                    # Select the paths from the current combination of data
                    # type, condition, stimulus, and repetition.
                    sel = (scanpath["datatype"]==di) & \
                        (scanpath["condition"]==ci) & \
                        (scanpath["stimulus"]==si) & \
                        (scanpath["repeat"]==ri)
                    # Plot the selected samples.
                    ax[ci,di].plot( \
                        scanpath["X_reduced_{}".format(method)][sel,0], \
                        scanpath["X_reduced_{}".format(method)][sel,1], \
                        ".", markersize=3, color=cmap(norm(ri+voffset)), \
                        alpha=1.0)
                    # Set the axis limit.
                    if method == "mds":
                        ax[ci,di].set_xlim([-10,10])
                        ax[ci,di].set_ylim([-10,10])
                    elif method == "umap":
                        ax[ci,di].set_xlim([-6,6.5])
                        ax[ci,di].set_ylim([-7.5,7])
                
            # Add axis titles.
            if ci == 0:
                ax[ci,di].set_title(datatype.capitalize(), \
                    fontsize=FONTSIZE["axtitle"])
            # Add axix labels.
            if ci == n_conditions-1:
                ax[ci,di].set_xlabel(r"{} C1 (au)".format(method.upper()), \
                    fontsize=FONTSIZE["label"])
        
        # Add the axis titles / labels for the y-axes.
        lbl = r"{} C2 (au)".format(method.upper())
        ax[ci,0].set_ylabel( \
            r"\fontsize{{{}pt}}{{3em}}\selectfont{{}}{{{}\\\\}}".format( \
            FONTSIZE["axtitle"], condition.capitalize()) + \
            r"{{\fontsize{{{}pt}}{{3em}}\selectfont{{}}{{{}}}}}".format( \
            FONTSIZE["label"], lbl))

        # Add the legend by plotting out-of-view samples.
        for ri in range(n_repeats):
            if ri == 0:
                lbl = r"1$^{st}$ presentation"
            else:
                lbl = r"Repeat {}".format(ri)
            x_off = ax[ci,-1].get_xlim()[0] - 100
            y_off = ax[ci,-1].get_ylim()[0] - 100
            ax[ci,-1].plot(x_off, y_off, "o", markersize=10, \
                color=cmap(norm(ri+voffset)), alpha=1.0, label=lbl)
        if method == "mds":
            leg_loc = "upper right"
        elif method == "umap":
            leg_loc = "lower left"
        else:
            log_loc = "best"
        ax[ci,-1].legend(loc=leg_loc, fontsize=FONTSIZE["legend"])
    
    # Remove ticklabels.
    for ci in range(ax.shape[0]):
        for di in range(ax.shape[1]):
            ax[ci,di].set_xticklabels([])
            ax[ci,di].set_yticklabels([])
        
    # Save figure.
    fig.savefig(os.path.join(OUTDIR, "path_reduction_{}.png".format(method)))
    pyplot.close(fig)



# # # # #
# CORRELATION MATRICES

corr = { \
    "stimuli":{}, \
    "presentations":{}, \
    }

# Loop through mouse and gaze.
for data_type in dwell.keys():
    
    # Create new entries in the correlation matrix dict.
    for key in corr.keys():
        corr[key][data_type] = {}

    # Loop through the conditions.
    for ci, condition in enumerate(CONDITIONS):
        
        # Compute the difference between neutral and affective stimulus. This
        # quantifies approach (positive values) or avoidance (negative values).
        # Computed as affective minus neutral, only for the current condition.
        # d has shape (n_participants, n_stimuli, n_presentations, n_bins)
        d = dwell[data_type][:,ci,:,:,0,:] - dwell[data_type][:,ci,:,:,1,:]
        
        # Average the dwell difference across all time bins.
        # d_m has shape (n_participants, n_stimuli, n_presentations)
        d_m = numpy.nanmean(d, axis=3)
        
        # Compute correlation matrices.
        for corr_type in corr.keys():
            # Choose the index number for this correlation matrix.
            if corr_type == "stimuli":
                d_m_i = numpy.nanmean(d_m, axis=2)
            elif corr_type == "presentations":
                d_m_i = numpy.nanmean(d_m, axis=1)
            # Create a new entry in the dict.
            corr[corr_type][data_type][condition] = \
                numpy.zeros((d_m_i.shape[1], d_m_i.shape[1]), dtype=float) \
                    * numpy.NaN
            # Walk through all combinations of stimuli or presentations.
            for j in range(d_m_i.shape[1]):
                for k in range(j, d_m_i.shape[1]):
                    if j == k:
                        r = 1.0
                    else:
                        r, p = scipy.stats.pearsonr(d_m_i[:,j], d_m_i[:,k])
                    corr[corr_type][data_type][condition][j,k] = r
                    corr[corr_type][data_type][condition][k,j] = r


# Compute differences in correlation values between the gaze and mouse
# correlation matrices.
corr_diff = {}
for corr_type in corr.keys():
    corr_diff[corr_type] = {}
    for condition in corr[corr_type]["mouse"].keys():
        # Transform r to z using Fisher z-transformation.
        r = numpy.copy(corr[corr_type]["gaze"][condition])
        r[r==1] = 0
        z_gaze = 0.5 * numpy.log((1+r)/(1-r))
        r = numpy.copy(corr[corr_type]["mouse"][condition])
        r[r==1] = 0
        z_mouse = 0.5 * numpy.log((1+r)/(1-r))
        # Count the numbers of participants in each.
        n_gaze = dwell["gaze"].shape[0]
        n_mouse = dwell["mouse"].shape[0]
        # Compute the standard deviation of the difference.
        sd = numpy.sqrt((1.0 / (n_gaze-3)) + (1.0 / (n_mouse-3)))
        # Compute the Z test statistic.
        z = (z_gaze - z_mouse) / sd
        # Compute p values for the z statistic.
        p = 2 * (1 - scipy.stats.norm.cdf(numpy.abs(z)))
        # Save the values.
        corr_diff[corr_type][condition] = { \
            "z":z, \
            "p":p, \
            }

# Loop through the different correlation types.
for corr_type in corr.keys():
    # Plot the correlation matrix in a 6-panel (2 rows, 3 columns) with
    # stimulus type (disgust, pleasant) in the rows, and measurement type
    # (gaze, mouse, difference) in the columns.
    fig, axes = pyplot.subplots(nrows=len(CONDITIONS), ncols=3, \
        figsize=(8.0*len(CONDITIONS),10.0), dpi=300)
    fig.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
        wspace=0.3, hspace=0.05)
    # Loop through the stimulus types.
    for ci, condition in enumerate(CONDITIONS):

        # First column: Gaze.
        # Second column: Mouse.
        data_types = corr[corr_type].keys()
        data_types.sort()
        for di, data_type in enumerate(data_types):
            # Convenience renaming.
            ax = axes[ci,di]
            m = corr[corr_type][data_type][condition]
            # Define plot specs.
            vmin = 0.0
            vmax = 1.0
            vstep = 0.2
            cmap = "viridis"
            # Plot the correlation matrix. This should contain only positive
            # values, so we only plot the range [0,1].
            ax.imshow(m, interpolation="none", aspect="equal", \
                origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
            # Annotate all values.
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if m[i,j] < 0:
                        val_str = str(round(m[i,j],2)).ljust(5, "0")
                    else:
                        val_str = str(round(m[i,j],2)).ljust(4, "0")
                    ax.annotate(val_str, (i-0.2,j-0.075), color="#eeeeec", \
                        fontsize=FONTSIZE["annotation"])
            # Add a colour bar.
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            bax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, \
                norm=norm, ticks=numpy.arange(vmin, vmax+vstep, vstep), \
                orientation='vertical')
            #cbar.set_ticklabels(bticklabels)
            cbar.set_label(r"Pearson $R$", fontsize=FONTSIZE["bar"])
            cbar.ax.tick_params(labelsize=FONTSIZE["ticklabels"])
            # Set the ticks and labels.
            if corr_type == "stimuli":
                lbl = "Stimulus number"
            elif corr_type == "presentations":
                lbl = "Presentation number"
            # Set axes' labels and ticks.
            if ci == 0:
                ax.set_title(data_type.capitalize(), \
                    fontsize=FONTSIZE["axtitle"])
            if ci == axes.shape[0] - 1:
                ax.set_xlabel(lbl, fontsize=FONTSIZE["label"])
            ax.set_xlim([-0.5, m.shape[1]-0.5])
            ax.set_xticks(range(0, m.shape[0]))
            ax.set_xticklabels(range(1, m.shape[1]+1), \
                fontsize=FONTSIZE["ticklabels"])
            if di == 0:
                ax.set_ylabel(lbl, fontsize=FONTSIZE["label"])
            ax.set_ylim([-0.5, m.shape[1]-0.5])
            ax.set_yticks(range(0, m.shape[0]))
            ax.set_yticklabels(range(1, m.shape[1]+1), \
                fontsize=FONTSIZE["ticklabels"])
            if di == 0:
                ax.set_ylabel( \
                    r"\fontsize{{{}pt}}{{3em}}\selectfont{{}}{{{}\\}}".format( \
                    FONTSIZE["axtitle"], condition.capitalize()) + \
                    r"{{\fontsize{{{}pt}}{{3em}}\selectfont{{}}{{{}}}}}".format( \
                    FONTSIZE["label"], lbl))
        
        # Final column: difference. Upper triangle = Z value, lower = p<0.05
        # Convenience renaming.
        ax = axes[ci,-1]
        z = corr_diff[corr_type][condition]["z"]
        p = corr_diff[corr_type][condition]["p"]
        # Plot specs.
        vmin = -3.0
        vmax = 3.0
        vstep = 1.0
        cmap = "BrBG"
        # Start with an empty matrix.
        m = numpy.zeros(z.shape, dtype=numpy.float32)
        # Set diagonal to 0.
        diag = numpy.diag_indices(m.shape[0], ndim=2)
        m[diag] = numpy.NaN
        # Set the upper triangle to Z values.
        ui = numpy.triu_indices(m.shape[0], k=1)
        m[ui] = z[ui]
        # Set the lower triangle to Boolean values reflecting p<0.05.
        li = numpy.tril_indices(m.shape[0], k=-1)
        p_ = numpy.copy(p)
        p_[z>=1] = vmax
        p_[z<=1] = vmin
        p_[p>=0.05] = 0.0
        m[li] = p_[li]
        # Plot the matrix.
        ax.imshow(m, interpolation="none", aspect="equal", \
            origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        # Annotate all values.
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if numpy.isnan(m[i,j]):
                    continue
                if i > j:
                    xy = (j-0.4,i-0.075)
                    fs = int(numpy.ceil(FONTSIZE["annotation"]*0.6))
                    if p[i,j] < 0.001:
                        val_str = r"$p<$0.001"
                    if p[i,j] < 0.05:
                        val_str = r"$p=$" + str(round(p[i,j],3)).ljust(5, "0")
                    else:
                        val_str = ""
                else:
                    fs = FONTSIZE["annotation"]
                    if m[i,j] < 0:
                        xy = (j-0.3,i-0.075)
                        val_str = str(round(m[i,j],2)).ljust(5, "0")
                    else:
                        xy = (j-0.2,i-0.075)
                        val_str = str(round(m[i,j],2)).ljust(4, "0")
                ax.annotate(val_str, xy, color="#eeeeec", fontsize=fs)
        # Add a colour bar.
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, \
            norm=norm, ticks=numpy.arange(vmin, vmax+vstep, vstep), \
            orientation='vertical')
        cbar.set_label(r"$Z$ statistic", fontsize=FONTSIZE["bar"])
        cbar.ax.tick_params(labelsize=FONTSIZE["ticklabels"])
        # Set the ticks and labels.
        if corr_type == "stimuli":
            lbl = "Stimulus number"
        elif corr_type == "presentations":
            lbl = "Presentation number"
        # Set axes' labels and ticks.
        if ci == 0:
            ax.set_title("Difference", fontsize=FONTSIZE["axtitle"])
        if ci == axes.shape[0] - 1:
            ax.set_xlabel(lbl, fontsize=FONTSIZE["label"])
        ax.set_xlim([-0.5, m.shape[1]-0.5])
        ax.set_xticks(range(0, m.shape[0]))
        ax.set_xticklabels(range(1, m.shape[1]+1), \
            fontsize=FONTSIZE["ticklabels"])
        if di == 0:
            ax.set_ylabel(lbl, fontsize=FONTSIZE["label"])
        ax.set_ylim([-0.5, m.shape[1]-0.5])
        ax.set_yticks(range(0, m.shape[0]))
        ax.set_yticklabels(range(1, m.shape[1]+1), \
            fontsize=FONTSIZE["ticklabels"])
    
    # Save the figure.
    fig.savefig(os.path.join(OUTDIR, "correlation_matrices_{}.png".format( \
        corr_type)))
    pyplot.close(fig)


# # # # #
# DWELL TIMES

# Create a four-panel figure. (Top row for lines, bottom row for heatmaps;
# columns for different conditions.)
fig, axes = pyplot.subplots(nrows=2, ncols=len(CONDITIONS), \
    figsize=(8.0*len(CONDITIONS),9.0), dpi=300.0)
fig.subplots_adjust(left=0.05, bottom=0.07, right=0.94, top=0.95,
    wspace=0.15, hspace=0.2)

# Compute the bin centres to serve as x-values.
time_bin_centres = bin_edges[:-1] + numpy.diff(bin_edges)/2.0

# Loop through the conditions.
for ci, condition in enumerate(CONDITIONS):
    
    # Choose the top-row, and the column for this condition.
    ax = axes[0,ci]
    
    # Compute the difference between neutral and affective stimulus. This
    # quantifies approach (positive values) or avoidance (negative values).
    # Computed as affective minus neutral, only for the current condition.
    # d has shape (n_participants, n_stimuli, n_presentations, n_bins)
    d_gaze = dwell["gaze"][:,ci,:,:,0,:] - dwell["gaze"][:,ci,:,:,1,:]
    d_mouse = dwell["mouse"][:,ci,:,:,0,:] - dwell["mouse"][:,ci,:,:,1,:]
    
    # Average over all stimuli, and convert to percentages.
    # d_gaze has shape (n_participants, n_presentations, n_bins)
    d_gaze = 100 * numpy.nanmean(d_gaze, axis=1)
    d_mouse = 100 * numpy.nanmean(d_mouse, axis=1)
    
    # Perform Welch's t-test, and compute Cohen's d for it.
    n1 = d_gaze.shape[0]
    m1 = numpy.mean(d_gaze, axis=0)
    sd1 = numpy.std(d_gaze, axis=0)

    n2 = d_mouse.shape[0]
    m2 = numpy.mean(d_mouse, axis=0)
    sd2 = numpy.std(d_mouse, axis=0)

    sd_pooled = numpy.sqrt((n1*(sd1**2) + n2*(sd2**2)) / float(n1+n2))
    cohen_d = (m1-m2) / sd_pooled
    df = ((((sd1**2) / n1) + ((sd2**2) / n2)) ** 2) / (((sd1**4) \
        / ((n1**2)*(n1-1))) + ((sd2**4) / ((n2**2)*(n2-1))))
    t, p = scipy.stats.ttest_ind(d_gaze, d_mouse, equal_var=False)

    # Average over all different stimuli.
    # m, sem, and ci_95 have shape (n_presentations, n_bins)
    m = m1 - m2
    # Standard error computed using Satterthwaite approximation, to avoid
    # assuming the variances of gaze and mouse are equal (they are not at all
    # points in time).
    sem = numpy.sqrt((sd1**2 / n1) + (sd2**2 / n2))
    ci_95 = 1.96 * sem
    
    # Linear regression with group membership as predictor. (Strictly speaking
    # this is equivalent to a Student's t-test, not Welch's.) We're doing this
    # to compute Bayesian Information Criterions for the difference predicted
    # by group (gaze vs mouse) membership, and for the null model (intercept
    # only). This allows us to compute a Bayes Factor from the BICs, and thus
    # gather evidence for the null.
    bf_01 = numpy.ones(p.shape, dtype=numpy.float32)
    bf_10 = numpy.ones(p.shape, dtype=numpy.float32)
    # Construct the x values for membership (0=gaze, 1=mouse).
    x = numpy.hstack([numpy.zeros(d_gaze.shape[0]), \
        numpy.ones(d_mouse.shape[0])])
    # Go through all samples (should be vectorised, but CBA).
    for i_pres in range(d_gaze.shape[1]):
        for i_bin in range(d_gaze.shape[2]):
            
            # Paste together the outcome values.
            y = numpy.hstack([d_gaze[:,i_pres,i_bin], d_mouse[:,i_pres,i_bin]])

            # Use statsmodels for the linear regression.
            if REGRESSION_BACKEND == "statsmodels":
                # Add a constant to the design matrix.
                x_ = numpy.ones((x.shape[0], 2), dtype=numpy.float32)
                x_[:,1] = x
                # Fit the linear regression with and without the group
                # membership predictor.
                m_alt = OLS(y, x_)
                result_alt = m_alt.fit()
                bic_1 = result_alt.bic
                # Fit the null model (only the intercept).
                m_nul = OLS(y, x_[:,0])
                result_nul = m_nul.fit()
                bic_0 = result_nul.bic
            
            # Fall back on standard SciPy approach.
            else:
                # Run a simple linear regression.
                b, b0, r, p_, err = scipy.stats.linregress(x, y)
                # Used this to double-check p value from linregress.
                #p_ = 2*(1 - scipy.stats.t.cdf(numpy.abs(b/err), df=y.shape[0]-1))
                # Compute residual sum of squares.
                y_pred = b0 + b*x
                ss_res = numpy.sum((y - y_pred)**2)
                # Compute the BIC for this model.
                n = x.shape[0]
                k = 2
                bic_1 = n + n*numpy.log(2*numpy.pi) + n*numpy.log(ss_res/n) + \
                    numpy.log(n)*(k+1)
    
                # The slope in a linear regression with ONLY a slope is equal to
                # the average of the outcome. We can thus compute the residual sum
                # of squares directly of the mean, because y_pred = mean(y)
                ss_res = numpy.sum((y - numpy.mean(y))**2)
                # Compute the BIC for the null model.
                n = x.shape[0]
                k = 1
                bic_0 = n + n*numpy.log(2*numpy.pi) + n*numpy.log(ss_res/n) + \
                    numpy.log(n)*(k+1)

            # Compute the Bayes Factor for null (BF01) and alternative (BF10)
            # hypotheses. While 1/BF10 == BF01, we'll use both later for the
            # visualisation of evidence for alternative or null.
            bf_01[i_pres,i_bin] = numpy.exp(0.5 * (bic_1 - bic_0))
            bf_10[i_pres,i_bin] = numpy.exp(0.5 * (bic_0 - bic_1))

    # Set Cohen's d to 0 where there is no effect.
    if MULTIPLE_COMPARISONS_CORRECTION is None:
        cohen_d[p > 0.05] = 0
    elif type(MULTIPLE_COMPARISONS_CORRECTION) == float:
        cohen_d[p > MULTIPLE_COMPARISONS_CORRECTION] = 0
    elif MULTIPLE_COMPARISONS_CORRECTION == "bonferroni":
        cohen_d[p > 0.05/p.shape[1]] = 0.0
    elif MULTIPLE_COMPARISONS_CORRECTION == "holm":
        for j in range(p.shape[0]):
            sorted_i = numpy.argsort(p[j,:])
            alpha = 0.05 / numpy.arange(p.shape[1],0.9,-1)
            cutoff = numpy.where(p[j,:][sorted_i] > alpha)[0][0]
            cohen_d[j,sorted_i[cutoff:]] = 0

    # Specify the colour map for the current condition.
    cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
    voffset = 3
    vmin = 0
    vmax = dwell["mouse"].shape[3] + voffset
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the separation line and background colours.
    ax.plot(time_bin_centres, numpy.zeros(time_bin_centres.shape), ':', lw=3, \
        color="black", alpha=0.5)
    ax.fill_between(time_bin_centres, \
        numpy.ones(time_bin_centres.shape)*YLIM["dwell_p"][0], \
        numpy.zeros(time_bin_centres.shape), color="black", alpha=0.05)
    annotate_x = time_bin_centres[0] + \
        0.02*(time_bin_centres[-1]-time_bin_centres[0])
    ax.annotate("Mouse avoidance higher", (annotate_x, YLIM["dwell_p"][1]-7), \
        fontsize=FONTSIZE["annotation"])
    ax.annotate("Gaze avoidance higher", (annotate_x, YLIM["dwell_p"][0]+3), \
        fontsize=FONTSIZE["annotation"])
    
    # LINES
    # Plot each stimulus presentation separately.
    for j in range(m.shape[0]):
        
        # Define the label.
        if j == 0:
            lbl = "First presentation"
        else:
            lbl = "Stimulus repeat {}".format(j)

        # PLOT THE LIIIIIIINE!
        ax.plot(time_bin_centres, m[j,:], "--", color=cmap(norm(j+voffset)), \
            lw=3, label=lbl)
        # Shade the confidence interval.
        ax.fill_between(time_bin_centres, m[j,:]-ci_95[j,:], \
            m[j,:]+ci_95[j,:], alpha=0.2, color=cmap(norm(j+voffset)))
        
    # Add a legend.
    if condition in LEGENDLOCS.keys():
        loc = LEGENDLOCS[condition]
    else:
        loc = "best"
    ax.legend(loc=loc, fontsize=FONTSIZE["legend"])

    # Finish the upper plot.
    ax.set_title(condition.capitalize(), fontsize=FONTSIZE["axtitle"])
    ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
    ax.set_xlim([0, TRIAL_DURATION])
    ax.set_xticks(range(0, TRIAL_DURATION+1, 1000))
    ax.set_xticklabels(range(0, (TRIAL_DURATION//1000 + 1)), \
        fontsize=FONTSIZE["ticklabels"])
    if ci == 0:
        ax.set_ylabel(r"$\Delta$dwell (gaze - mouse)", \
            fontsize=FONTSIZE["label"])
    ax.set_ylim(YLIM["dwell_p"])
    ax.set_yticks(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10))
    ax.set_yticklabels(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10), \
        fontsize=FONTSIZE["ticklabels"])
    
    # HEATMAPS
    # Choose the heatmap row, and the column for the current condition.
    ax = axes[1,ci]

    # Create the colourmap for this condition.
    cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS["heatmap"])
    vmin = YLIM["bayes"][0]
    vmax = YLIM["bayes"][1]
    vstep = numpy.min(numpy.abs(YLIM["bayes"])) // 2
    bticks = range(vmin, vmax+1, vstep)
    bticklabels = map(str, bticks)

    # Plot the heatmap.
    heatmap = numpy.log(bf_10)
    ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", \
        aspect="auto", origin="upper")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    bax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
        ticks=bticks, orientation='vertical')
    cbar.set_ticklabels(bticklabels)
    if ci == axes.shape[1]-1:
        cbar.set_label(r"$\log(BF_{10})$" + "\n" \
            r"$h_{0} \Leftarrow$ evidence $\Rightarrow h_{1}$", \
            fontsize=FONTSIZE["bar"])
    cbar.ax.tick_params(labelsize=FONTSIZE["ticklabels"])

    #ax.set_title(CONDITIONS[condition], fontsize=FONTSIZE["axtitle"])
    ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
    ax.set_xlim([-0.5, m.shape[1]-0.5])
    ax.set_xticks(numpy.linspace(-0.5, m.shape[1]+0.5, \
        num=TRIAL_DURATION//1000 + 1))
    ax.set_xticklabels(range(0, (TRIAL_DURATION//1000 + 1)), \
        fontsize=FONTSIZE["ticklabels"])
    if ci == 0:
        ax.set_ylabel("Presentation number", fontsize=FONTSIZE["label"])
    ax.set_ylim([NTRIALSPERSTIM-0.5, -0.5])
    ax.set_yticks(range(0, NTRIALSPERSTIM))
    ax.set_yticklabels(range(1, NTRIALSPERSTIM+1), \
        fontsize=FONTSIZE["ticklabels"])

# Save the figure.
fig.savefig(os.path.join(OUTDIR, "dwell_percentages_difference.png"))
pyplot.close(fig)
