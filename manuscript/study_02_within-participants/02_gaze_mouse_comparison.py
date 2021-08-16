#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import copy

import cv2
import numpy
import scipy.stats

import pandas
import matplotlib
from matplotlib import pyplot
from matplotlib import patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.api import add_constant, OLS
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tools.eval_measures import aic, bic
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
DO_RATINGS = True

# Comparison approach.
# There are two possible approaches to this. The first is to simply compute
# the difference in dwell proportion between affective and neutral stimulus.
# The downside of this, is that this difference is confounded with missing
# data. The second approach is to compute what proportion of non-other gaze
# was for the affective stimulus, i.e.: affect / (affect+neutral).
# "difference": affect - neutral
# "proportional": affect / (affect + neutral)
COMPARE_APPROACH = "difference"

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
VIEWBINS = (VIEWPORT[0]//30, VIEWPORT[1]//30) #(VIEWPORT[0]//40, VIEWPORT[1]//40)
STIMRECT = { \
    "affective": (401,  289, 480, 360), \
    "neutral":   (1038, 289, 480, 360), \
    }
# Within-participant model with and without data type ("method", i.e. gaze or
# mouse). The null model is without the method; comparing it against a model
# with method should give some insight into how different the methods were.
WITHIN_MODEL = { \
    "h1": "dwell ~ method", \
    "h0": "dwell ~ 1", \
    }
# Define which variables are categorical.
LME_CATEGORICAL_VARIABLES = ["method", "condition", "stimulus_nr", \
    "presentation_nr"]
# Set the variables that need to be standardised. (all non-categorical ones.)
LME_STANDARDISE_THESE = ["dwell", "stimulus_rating"]

# FILES AND FOLDERS.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data", "processed_data")
LONGDATADIR = os.path.join(DIR, "data", "processed_data", "dwell_long_per_bin")
PARTICIPANTPATH = os.path.join(DATADIR, "participants_{}_memmap.dat")
RATINGSPATH = os.path.join(DATADIR, "ratings_{}_memmap.dat")
RATINGSSHAPEPATH = os.path.join(DATADIR, "ratings_shape_{}_memmap.dat")
DWELLPATH = os.path.join(DATADIR, "dwell_mouse_memmap.dat")
DWELLSHAPEPATH = os.path.join(DATADIR, "dwell_mouse_shape_memmap.dat")
GAZEPATH = os.path.join(DATADIR, "dwell_gaze_memmap.dat")
GAZESHAPEPATH = os.path.join(DATADIR, "dwell_gaze_shape_memmap.dat")
RESAMPLEPATH = os.path.join(DATADIR, "resampled_xy_{}.dat")
RESAMPLESHAPEPATH = os.path.join(DATADIR, "resampled_xy_shape_{}.dat")
MDSPATH = os.path.join(DATADIR, "MDS_reduced_xy.dat")
UMAPPATH = os.path.join(DATADIR, "UMAP_reduced_xy.dat")
BAYESPATH = os.path.join(DATADIR, "bayes_factors.dat")
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
    "welch":  [-2, 2], \
    "bayes":  [-5, 5], \
    }

# Set the locations of legends.
LEGENDLOCS = { \
    "disgust":  "lower right", \
    "pleasant": "lower right", \
    }


# # # # #
# LOAD DATA

# PARTICIPANT CODES
participants = {}
for data_type in ["gaze", "mouse"]:
    pp_memmap = numpy.memmap(PARTICIPANTPATH.format(data_type), dtype="|S32", \
        mode="r")
    participants[data_type] = list(pp_memmap)
# Create a list of participants who have data for both gaze and mouse.
full_sets = []
for ppname in participants["gaze"]:
    if ppname in participants["mouse"]:
        full_sets.append(ppname)

# RATINGS
if DO_RATINGS:
    ratings_ = {}
    
    # Load rating data.
    for data_type in ["gaze", "mouse"]:
        ratings_shape = tuple(numpy.memmap(RATINGSSHAPEPATH.format(data_type), \
            dtype=numpy.int16, mode="r"))
        ratings_[data_type] = numpy.memmap(RATINGSPATH.format(data_type), \
            dtype=numpy.float32, mode="r", shape=ratings_shape)

# Only use full datasets.
ratings_shape = numpy.copy(ratings_["gaze"].shape)
ratings_shape[0] = len(full_sets)
ratings = {}
# Loop through the data types and conditions.
for di, data_type in enumerate(["gaze", "mouse"]):
    ratings[data_type] = numpy.zeros(ratings_shape, dtype=numpy.float32)
    # Run through all participants.
    for pi, ppname in enumerate(full_sets):
        # Get the participant index number for this datatype.
        ppi = participants[data_type].index(ppname)
        # Put their data in the right place.
        ratings[data_type][pi,:,:] = ratings_[data_type][ppi,:,:]

# DWELL TIMES
dwell_ = {}

# Load the dwell data's shape from the dwell_shape file.
gaze_shape = tuple(numpy.memmap(GAZESHAPEPATH, dtype=numpy.int32, mode="r"))
# Load the dwell data from file.
dwell_["gaze"] = numpy.memmap(GAZEPATH, dtype=numpy.float32, mode="r", \
    shape=gaze_shape)

# Load the dwell data's shape from the dwell_shape file.
mouse_shape = tuple(numpy.memmap(DWELLSHAPEPATH, dtype=numpy.int32, mode="r"))
# Load the dwell data from file.
dwell_["mouse"] = numpy.memmap(DWELLPATH, dtype=numpy.float32, mode="r", \
    shape=mouse_shape)

# Only use full datasets.
dwell_shape = numpy.copy(dwell_["gaze"].shape)
dwell_shape[0] = len(full_sets)
dwell = {}
# Loop through the data types and conditions.
for di, data_type in enumerate(["gaze", "mouse"]):
    dwell[data_type] = numpy.zeros(dwell_shape, dtype=numpy.float32)
    # Run through all participants.
    for pi, ppname in enumerate(full_sets):
        # Get the participant index number for this datatype.
        ppi = participants[data_type].index(ppname)
        # Put their data in the right place.
        dwell[data_type][pi,:,:,:,:] = dwell_[data_type][ppi,:,:,:,:]

# Recompute the bin edges.
bin_edges = numpy.arange(0, TRIAL_DURATION+BINWIDTH/2, BINWIDTH, \
    dtype=numpy.float32)

# SCAN PATHS
scan_ = {}
# Load scan paths.
for data_type in ["mouse", "gaze"]:
    # The shape is (n_participants, x=0/y=1, n_conditions, n_stimuli,
    # n_presentations, n_samples)
    scan_shape = tuple(numpy.memmap(RESAMPLESHAPEPATH.format(data_type), \
        dtype=numpy.int32, mode="r"))
    scan_[data_type] = numpy.memmap(RESAMPLEPATH.format(data_type), \
        dtype=numpy.float32, mode="r", shape=scan_shape)

# Only use full datasets.
scan_shape = numpy.copy(scan_["gaze"].shape)
scan_shape[0] = len(full_sets)
scan = {}
# Loop through the data types and conditions.
for di, data_type in enumerate(["gaze", "mouse"]):
    scan[data_type] = numpy.zeros(scan_shape, dtype=numpy.float32)
    # Run through all participants.
    for pi, ppname in enumerate(full_sets):
        # Get the participant index number for this datatype.
        ppi = participants[data_type].index(ppname)
        # Put their data in the right place.
        scan[data_type][pi,:,:,:,:,:] = scan_[data_type][ppi,:,:,:,:,:]

# Overwrite participants, now that we're only using full datasets.
participants = {}
for data_type in ["gaze", "mouse"]:
    participants[data_type] = full_sets


# # # # #
# DATA LONG FORMAT

# Export the data in a long format, so that it can be used for LME.
with open(os.path.join(DATADIR, "dwell_long.csv"), "w") as f:
    header = ["subject", "method", "condition", "stimulus_nr", \
        "presentation_nr", "stimulus_rating", "dwell"]
    f.write(",".join(map(str, header)))
    for ppname in full_sets:
        # Loop through both data types.
        ppi = {}
        for datatype in ["gaze", "mouse"]:
            # Get the participant index for this datatype.
            ppi[datatype] = participants[datatype].index(ppname)
            # Loop through all conditions, stimuli, and presentations.
            for ci in range(dwell[datatype].shape[1]):
                condition = CONDITIONS[ci]
                for si in range(dwell[datatype].shape[2]):
                    rating = ratings[datatype][ppi[datatype],ci,si]
                    for pi in range(dwell[datatype].shape[3]):
                        if COMPARE_APPROACH == "difference":
                            d = numpy.nanmean( \
                                dwell[datatype][ppi[datatype],ci,si,pi,0,:] - \
                                dwell[datatype][ppi[datatype],ci,si,pi,1,:])
                        elif COMPARE_APPROACH == "proportional":
                            d = numpy.nanmean( \
                                dwell[datatype][ppi[datatype],ci,si,pi,0,:] / \
                                (dwell[datatype][ppi[datatype],ci,si,pi,0,:] + \
                                dwell[datatype][ppi[datatype],ci,si,pi,1,:]))
                        line = [ppi[datatype], datatype, condition, si+1, \
                            pi+1, rating, d]
                        f.write("\n" + ",".join(map(str, line)))

# Export the data in a long format for each bin, so that it can be used for a
# series of LMEs (to be used to compare stimulus and null model.
if not os.path.isdir(LONGDATADIR):
    os.mkdir(LONGDATADIR)
for ci in range(dwell[datatype].shape[1]):
    condition = CONDITIONS[ci]
    for bi in range(dwell["gaze"].shape[5]):
        with open(os.path.join(LONGDATADIR, \
            "dwell_long_bin_{}_{}.csv".format(condition, bi)), "w") as f:
            header = ["subject", "method", "condition", "stimulus_nr", \
                "presentation_nr", "stimulus_rating", "dwell"]
            f.write(",".join(map(str, header)))
            for ppname in full_sets:
                # Loop through both data types.
                ppi = {}
                for datatype in ["gaze", "mouse"]:
                    # Get the participant index for this datatype.
                    ppi[datatype] = participants[datatype].index(ppname)
                    # Loop through all conditions, stimuli, and presentations.
                    for si in range(dwell[datatype].shape[2]):
                        rating = ratings[datatype][ppi[datatype],ci,si]
                        for pi in range(dwell[datatype].shape[3]):
                            if COMPARE_APPROACH == "difference":
                                d = dwell[datatype][ppi[datatype],ci,si,pi,0,bi] - \
                                    dwell[datatype][ppi[datatype],ci,si,pi,1,bi]
                            elif COMPARE_APPROACH == "proportional":
                                d = dwell[datatype][ppi[datatype],ci,si,pi,0,bi] / \
                                    (dwell[datatype][ppi[datatype],ci,si,pi,0,bi] + \
                                    dwell[datatype][ppi[datatype],ci,si,pi,1,bi])
                            line = [ppi[datatype], datatype, condition, si+1, \
                                pi+1, rating, d]
                            f.write("\n" + ",".join(map(str, line)))


# # # # #
# RATING CORRELATIONS

if DO_RATINGS:

    # Compute gaze differences.
    d = {}
    for key in dwell.keys():
        if COMPARE_APPROACH == "difference":
            d[key] = numpy.nanmean(dwell[key][:,:,:,:,0,:] - \
                dwell[key][:,:,:,:,1,:], axis=4)
        elif COMPARE_APPROACH == "proportional":
            d[key] = numpy.nanmean(dwell[key][:,:,:,:,0,:] / \
                (dwell[key][:,:,:,:,0,:] + dwell[key][:,:,:,:,1,:]), axis=4)
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
            ax[ci,di].plot(x[:,1], y_pred, "-", lw=3, color=col, alpha=1.0, \
                label=lbl)
            # Plot regression confidence envelope.
            ax[ci,di].fill_between(x[:,1], predict_mean_ci_low, \
                predict_mean_ci_upp, color=col, alpha=0.3)
            # Plot prediction intervals.
            ax[ci,di].plot(x[:,1], predict_ci_low, '-', lw=2, color=col, \
                alpha=0.5)
            ax[ci,di].plot(x[:,1], predict_ci_upp, '-', lw=2, color=col, \
                alpha=0.5)
            
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
            if COMPARE_APPROACH == "difference":
                ax[ci,di].set_ylim([-100,100])
            elif COMPARE_APPROACH == "proportional":
                ax[ci,di].set_ylim([0,100])
            # Set labels.
            ax[ci,di].set_xticklabels(ax[ci,di].get_xticks(), \
                fontsize=FONTSIZE["ticklabels"])
            ax[ci,di].set_yticklabels(ax[ci,di].get_yticks(), \
                fontsize=FONTSIZE["ticklabels"])
            
            # Add a legend.
            ax[ci,di].legend(loc="upper right", fontsize=FONTSIZE["legend"])
            
        # Set the y label.
        if COMPARE_APPROACH == "difference":
            ax[ci,0].set_ylabel(r"$\Delta$dwell time ({}-neutral \% pt.)".format( \
                condition), fontsize=FONTSIZE["label"])
        elif COMPARE_APPROACH == "proportional":
            ax[ci,0].set_ylabel(r"Dwell time ({} \%)".format( \
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

# GAZE*MOUSE CORRELATION
# Two rows.     Top: Overall correlation between gaze and mouse avoidance.
#               Bottom: Gaze*mouse correlation for all time bins.
# Two columns.  Left: Disgust.
#               Right: Pleasant.

# Compute the dwell time differences (averaged over stimuli and presentations)
# per time bin for each data type, aligned so that participants match up.
shape = len(full_sets), 2, len(CONDITIONS), dwell["gaze"].shape[5]
d = numpy.zeros(shape, dtype=numpy.float32)
# Loop through the data types and conditions.
for di, data_type in enumerate(["gaze", "mouse"]):
    for ci, condition in enumerate(CONDITIONS):
        # Run through all participants.
        for pi, ppname in enumerate(full_sets):
            # Get the participant index number for this datatype.
            ppi = participants[data_type].index(ppname)
            # Compute the difference between neutral and affective stimulus. This
            # quantifies approach (positive values) or avoidance (negative values).
            # Computed as affective minus neutral, only for the current condition.
            # dwell has shape (n_participants, n_conditions, n_stimuli, 
            #   n_presentations, n_bins)
            # d has shape (n_participants, n_datatypes, n_conditions, n_bins)
            # Average over stimuli (first) and presentations (second).
            # Pure difference:
            # dwell(affective) - dwell(neutral)
            if COMPARE_APPROACH == "difference":
                d[pi,di,ci,:] = numpy.nanmean(numpy.nanmean( \
                    dwell[data_type][ppi,ci,:,:,0,:] - \
                    dwell[data_type][ppi,ci,:,:,1,:], axis=1), axis=0)
            # Proportion of non-missing dwell on the affective stimulus.
            # dwell(affective) / (dwell(affective) + dwell(neutral))
            elif COMPARE_APPROACH == "proportional":
                d[pi,di,ci,:] = numpy.nanmean(numpy.nanmean( \
                    dwell[data_type][ppi,ci,:,:,0,:] / \
                    (dwell[data_type][ppi,ci,:,:,0,:] + \
                    dwell[data_type][ppi,ci,:,:,1,:]), axis=1), axis=0)
            else:
                raise Exception("Compare approach {} not known.".format( \
                    COMPARE_APPROACH))
# Transform to percentages.
d *= 100

# Plot the correlations.
fig, axes = pyplot.subplots(nrows=len(CONDITIONS), ncols=2, \
    figsize=(7.0*len(CONDITIONS), 10.0), dpi=300)
fig.subplots_adjust(left=0.07, bottom=0.06, right=0.95, top=0.95,
    wspace=0.25, hspace=0.2)
# Loop through the conditions.
for ci, condition in enumerate(CONDITIONS):
    
    # Specify the colour map for the current condition.
    cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
    voffset = 3
    vmin = 0
    vmax = 4
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # Grab just a single colour.
    col = cmap(norm(voffset))

    # PANEL 1: Correlation of average dwell difference.
    ax = axes[0,ci]
    ax.set_title(condition.capitalize(), fontsize=FONTSIZE["axtitle"])
    # Find the order of all x-values, for plotting purposes.
    # NOTE: x will be used to order, and y values will be ordered
    # according to x's order. Hence, the correct values will remain
    # linked. (Separate ordering would be exceptionally wrong.)
    d_avg = numpy.nanmean(d, axis=3)
    notnan = (numpy.isnan(d_avg[:,0,ci])==False) & \
        (numpy.isnan(d_avg[:,1,ci])==False)
    order = numpy.argsort(d_avg[:,0,ci][notnan])
    x = add_constant(d_avg[:,0,ci][notnan][order])
    y = d_avg[:,1,ci][notnan][order]

    # Compute the Pearson correlation.
    r, p = scipy.stats.pearsonr(x[:,1], y)
    # Construct a label from the correlation outcome.
    if r < 0:
        r_str = "$R=${}".format(str(round(r, 2)).ljust(5, '0'))
    else:
        r_str = "$R=${}".format(str(round(r, 2)).ljust(4, '0'))
    if p < 0.001:
        p_str = r"$p<$0.001"
    else:
        p_str = "$p=${}".format(str(round(p, 3)).ljust(5, '0'))
    lbl = r"{}, {}".format(r_str, p_str)

    # Plot the individual samples.
    ax.plot(x[:,1], y, "o", color=col, alpha=0.3)
    
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
    ax.plot(x[:,1], y_pred, "-", lw=3, color=col, alpha=1.0, label=lbl)
    # Plot regression confidence envelope.
    ax.fill_between(x[:,1], predict_mean_ci_low, \
        predict_mean_ci_upp, color=col, alpha=0.3)
    
    # Set the axis limits.
    if COMPARE_APPROACH == "difference":
        lim = [-100, 100]
    elif COMPARE_APPROACH == "proportional":
        lim = [0, 100]
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # Set the axis ticks.
    xticks = numpy.round(numpy.linspace(lim[0], lim[1], 11)).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(map(str, xticks), fontsize=FONTSIZE["ticklabels"])
    yticks = numpy.round(numpy.linspace(lim[0], lim[1], 11)).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(map(str, yticks), fontsize=FONTSIZE["ticklabels"])

    # Set the axis labels.
    if COMPARE_APPROACH == "difference":
        ax.set_xlabel(r"Gaze $\Delta$dwell time (\% pt.)", \
            fontsize=FONTSIZE["label"])
        ax.set_ylabel(r"Mouse $\Delta$dwell time (\% pt.)", \
            fontsize=FONTSIZE["label"])
    elif COMPARE_APPROACH == "proportional":
        ax.set_xlabel(r"Gaze dwell time (\%)", \
            fontsize=FONTSIZE["label"])
        ax.set_ylabel(r"Mouse dwell time (\%)", \
            fontsize=FONTSIZE["label"])
    # Add a legend.
    ax.legend(loc="lower right", fontsize=FONTSIZE["legend"])
    

    # PANEL 2: Correlation over time.
    ax = axes[1,ci]
    # Compute the Pearson correlation over time.
    x = d[:,0,ci,:]
    y = d[:,1,ci,:]
    m_x = numpy.nanmean(x, axis=0)
    m_y = numpy.nanmean(y, axis=0)
    r = numpy.nansum((x-m_x) * (y-m_y), axis=0) / \
        numpy.sqrt(numpy.nansum((x-m_x)**2, axis=0) * \
        numpy.nansum((y-m_y)**2, axis=0))
    # Compute the standard error of the correlation over time.
    df = d.shape[0] - 2
    se_r = numpy.sqrt((1 - r**2) / df)
    
    # Plot the correlation over time.
    t = (bin_edges[:-1] + numpy.diff(bin_edges)/2.0) / 1000.0
    ax.plot(t, r, "-", lw=3, color=col, alpha=1.0)
    ax.fill_between(t, r-se_r, r+se_r, color=col, alpha=0.3)
    
    # Compute the critical value, to highlight in the plot.
    alpha = 0.05
    crit_t = scipy.stats.t.ppf(alpha/2.0, df=df)
    crit_r = numpy.sqrt((crit_t**2) / (crit_t**2 + df))
    # Plot the critical values, and grey-out the space between them.
    r_lo = -crit_r*numpy.ones(t.shape, dtype=numpy.float32)
    r_hi = crit_r*numpy.ones(t.shape, dtype=numpy.float32)
    ax.plot(t, r_lo, ":", lw=2, color="#000000", alpha=0.5)
    ax.plot(t, r_hi, ":", lw=2, color="#000000", alpha=0.5)
    ax.fill_between(t, r_lo, r_hi, color="#000000", alpha=0.1)
    
    # Set axis limits.
    ax.set_xlim([bin_edges[0]/1000.0, bin_edges[-1]/1000.0])
    ax.set_ylim([-1.0, 1.0])
    
    # Set tick labels.
    xticks = (numpy.round(numpy.linspace(bin_edges[0], bin_edges[-1], 11)) \
        / 1000.0).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(map(str, xticks), fontsize=FONTSIZE["ticklabels"])
    yticks = numpy.round(numpy.linspace(-1, 1, 11), 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(map(str, yticks), fontsize=FONTSIZE["ticklabels"])
    
    # Set axis labels.
    ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
    ax.set_ylabel("Gaze*Mouse correlation", fontsize=FONTSIZE["label"])
    
    # Annotate the meaning.
    annotate_x = t[0] + 0.02*(t[-1]-t[0])
    ax.annotate("Gaze and mouse postively related", (annotate_x, 0.85), \
        fontsize=FONTSIZE["annotation"])
    ax.annotate("No relationship between gaze and mouse", \
        (annotate_x, -crit_r+0.05), fontsize=FONTSIZE["annotation"])
    ax.annotate("Gaze and mouse negatively related", (annotate_x, -0.95), \
        fontsize=FONTSIZE["annotation"])

# Save the figure.
fig.savefig(os.path.join(OUTDIR, "dwell_correlations.png"))
pyplot.close(fig)


# TEST-RETEST
# Start with an empty dict to contain all different types of reliability.
#   "presentations" is for the Harris ICC across all presentations of the same 
#       stimulus.
#   "stimuli" is for the Harris ICC across all stimuli, separated by
#       presentation number.
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
        if COMPARE_APPROACH == "difference":
            d = dwell[data_type][:,ci,:,:,0,:] - dwell[data_type][:,ci,:,:,1,:]
        elif COMPARE_APPROACH == "proportional":
            d = dwell[data_type][:,ci,:,:,0,:] / \
                (dwell[data_type][:,ci,:,:,0,:]+dwell[data_type][:,ci,:,:,1,:])
        
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
                    txt = ax.annotate(val_str, (i-0.2,j-0.075), color="#eeeeec", \
                        fontsize=FONTSIZE["annotation"])
                    txt.set_path_effects([patheffects.withStroke(linewidth=2, \
                        foreground="#2e3436")])
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
                txt = ax.annotate(val_str, xy, color="#eeeeec", fontsize=fs)
                txt.set_path_effects([patheffects.withStroke(linewidth=2, \
                    foreground="#2e3436")])
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

# Load existing Bayes Factors for comparison of the gaze and mouse methods.
if os.path.isfile(BAYESPATH):
    print("\nLoading pre-computed gaze-mouse Bayes Factors.")
    bf_10 = numpy.memmap(BAYESPATH, dtype=numpy.float32, mode="r", \
        shape=(dwell["gaze"].shape[1], dwell["gaze"].shape[3], \
        dwell["gaze"].shape[5]))
# Compute Bayes Factors for comparison of the gaze and mouse methods.
else:
    print("\nComputing gaze-mouse Bayes Factors.")
    # Ignore the convergence warnings.
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
    # Create a new memory-mapped array to store data in.
    bf_10 = numpy.memmap(BAYESPATH, dtype=numpy.float32, mode="w+", \
        shape=(dwell["gaze"].shape[1], dwell["gaze"].shape[3], \
        dwell["gaze"].shape[5]))
    bf_10[:] = 1.0
    # Run through all conditions and bins.
    import time
    t = []
    for ci in range(bf_10.shape[0]):
        condition = CONDITIONS[ci]
        for bi in range(bf_10.shape[2]):
            print("\tBFs for {}, bin {}; avg iteration time {} seconds".format( \
                condition, bi, round(numpy.mean(t), 2)))
            # Load data.
            file_path = os.path.join(LONGDATADIR, \
                "dwell_long_bin_{}_{}.csv".format(condition, bi))
            data = pandas.read_csv(file_path)
            n_original = len(numpy.unique(data["subject"]))
            # Cast categorical values as string.
            for var in LME_CATEGORICAL_VARIABLES:
                data[var] = data[var].astype(str)
            for var in LME_STANDARDISE_THESE:
                if var in data.keys():
                    m = numpy.nanmean(data[var])
                    sd = numpy.nanstd(data[var])
                    data[var] = (data[var] - m) / sd
            for pi in range(bf_10.shape[1]):
                # Record the starting time for this iteration.
                t0 = time.time()
                # Select a subset of the data.
                data_ = data[data.presentation_nr==str(pi+1)]
                # Fit the method and its null model.
                lme_h1 = MixedLM.from_formula(WITHIN_MODEL["h1"], \
                    groups=data_["subject"], data=data_, missing="drop")
                lme_h1 = lme_h1.fit()
                lme_h0 = MixedLM.from_formula(WITHIN_MODEL["h0"], \
                    groups=data_["subject"], data=data_, missing="drop")
                lme_h0 = lme_h0.fit()
                # Compute the BICs.
                bic_h1 = bic(lme_h1.llf, lme_h1.nobs, lme_h1.df_modelwc)
                bic_h0 = bic(lme_h0.llf, lme_h0.nobs, lme_h0.df_modelwc)
                # Compute the Bayes Factor.
                bf_10[ci,pi,bi] = numpy.exp(0.5 * (bic_h0 - bic_h1))
                # Record the time of this iteration.
                t.append(time.time()-t0)
    # Save to disk.
    bf_10.flush()

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
    if COMPARE_APPROACH == "difference":
        d_gaze = dwell["gaze"][:,ci,:,:,0,:] - dwell["gaze"][:,ci,:,:,1,:]
        d_mouse = dwell["mouse"][:,ci,:,:,0,:] - dwell["mouse"][:,ci,:,:,1,:]
    elif COMPARE_APPROACH == "proportional":
        d_gaze = dwell["gaze"][:,ci,:,:,0,:] / \
            (dwell["gaze"][:,ci,:,:,0,:] + dwell["gaze"][:,ci,:,:,1,:])
        d_mouse = dwell["mouse"][:,ci,:,:,0,:] / \
            (dwell["mouse"][:,ci,:,:,0,:] + dwell["mouse"][:,ci,:,:,1,:])
    
    # Average over all stimuli, and convert to percentages.
    # d_gaze has shape (n_participants, n_presentations, n_bins)
    d_gaze = 100 * numpy.nanmean(d_gaze, axis=1)
    d_mouse = 100 * numpy.nanmean(d_mouse, axis=1)
    d_ = d_gaze - d_mouse
    
    # Perform a related-samples t-test for the difference.
    t, p = scipy.stats.ttest_rel(d_gaze, d_mouse, nan_policy="omit")

    # m, sem, and ci_95 have shape (n_presentations, n_bins)
    # Average over all different participants.
    m = numpy.nanmean(d_, axis=0)
    # Standard error computed using Satterthwaite approximation, to avoid
    # assuming the variances of gaze and mouse are equal (they are not at all
    # points in time).
    # new value = old value – subject average + grand average
    subj_avg = numpy.nanmean(numpy.nanmean(d_, axis=1), axis=1)
    subj_avg = subj_avg.reshape(subj_avg.shape[0],1)
    grand_avg = numpy.nanmean(d_, axis=None)
    # Can't do this vectorised due to confusion about which axes to use.
    #nv = d_ - subj_avg + grand_avg
    nv = numpy.copy(d_)
    for i in range(nv.shape[1]):
        nv[:,i,:] = d_[:,i,:] - subj_avg + grand_avg
    sem = numpy.nanstd(nv, axis=0) / numpy.sqrt(d_.shape[0])
    ci_95 = 1.96 * sem

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
    heatmap = numpy.log(bf_10[ci,:,:])
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

