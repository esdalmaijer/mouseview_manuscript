#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy

import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_1samp
import statsmodels.api
from statsmodels.stats.outliers_influence import summary_table

from reliability import cronbach_alpha, split_half, test_retest
from MouseViewParser.readers import gorilla
from shitty_gazepoint_files_parser import read_opengaze

pyplot.rc('text', usetex=True)


# # # # #
# CONSTANTS

# BASIC CONTROLS
# Type of data to load.
DATATYPE = "gaze"
# Specify the file name(s). We only have one here, but if you ran more than
# one batch, just chuck 'em all in here.
FILENAMES = ["data_exp_26934-v6_task-3wqj.csv"]
# Ratings are the same file, as participants did those before participating
# in both the gaze and the mouseview experiment.
RATINGFILENAME = { \
    "mouse":    "MouseView_Exp3_Ratings.csv", \
    "gaze":     "MouseView_Exp3_Ratings.csv", \
    }
# Overwrite the temporary data?
OVERWRITE_TMP = True
# Are rating data available?
DO_RATINGS = True

# EXCLUDED PARTICIPANTS
# The following participant numbers should be excluded due to being test runs
# by the researchers.
EXCLUDE = ["0", "9000"]

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

# GAZE DATA SETTINGS
# Image onset in trial.
GAZE_START = "IMAGE_START"
GAZE_IMAGE_END = "ITI_START"
GAZE_STOP = "STOP_TRIAL"

# ANALYSIS SETTINGS
# Bin size for re-referencing samples to bins across the trial duration.
# This is in milliseconds.
BINWIDTH = 100.0 / 3.0
# Number of samples to resample the (x,y) coordinates to.
N_RESAMPLES = 300
# Multiple comparisons correction.
MULTIPLE_COMPARISONS_CORRECTION = 0.05 #"no correction"

# FILES AND FOLDERS.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data", DATATYPE)
TMPDATADIR = os.path.join(DIR, "data", "processed_data")
if not os.path.isdir(TMPDATADIR):
    os.mkdir(TMPDATADIR)
FILEPATHS = []
for fname in FILENAMES:
    FILEPATHS.append(os.path.join(DATADIR, fname))
RATINGFILEPATH = os.path.join(DATADIR, RATINGFILENAME[DATATYPE])
PARTICIPANTPATH = os.path.join(TMPDATADIR, \
    "participants_{}_memmap.dat".format(DATATYPE))
RATINGSPATH = os.path.join(TMPDATADIR, \
    "ratings_{}_memmap.dat".format(DATATYPE))
RATINGSSHAPEPATH = os.path.join(TMPDATADIR, \
    "ratings_shape_{}_memmap.dat".format(DATATYPE))
DWELLPATH = os.path.join(TMPDATADIR, "dwell_mouse_memmap.dat")
DWELLSHAPEPATH = os.path.join(TMPDATADIR, "dwell_mouse_shape_memmap.dat")
GAZEPATH = os.path.join(TMPDATADIR, "dwell_gaze_memmap.dat")
GAZESHAPEPATH = os.path.join(TMPDATADIR, "dwell_gaze_shape_memmap.dat")
RESAMPLEPATH = os.path.join(TMPDATADIR, "resampled_xy_{}.dat".format(DATATYPE))
RESAMPLESHAPEPATH = os.path.join(TMPDATADIR, \
    "resampled_xy_shape_{}.dat".format(DATATYPE))
OUTDIR = os.path.join(DIR, "output")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# PLOTTING
PLOTCOLMAPS = { \
    "disgust":  "Oranges", \
    "pleasant": "Blues", \
    "heatmap":  "PiYG_r", \
    "t_vals":   "PiYG_r", \
    "bayes":    "BrBG", \
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
    "dwell_p":  [-60, 60], \
    "heatmap":  [-60, 60], \
    "t_vals":   [-14, 14], \
    "bayes":    [-5, 5], \
    }
# Set the locations of legends.
LEGENDLOCS = { \
    "disgust":  "upper right", \
    "pleasant": "lower right", \
    }


# # # # #
# BINNED RESAMPLING

def resample(x, n_resamples):
    
    # Number of samples (in original data) per bin (in resamples data).
    spb = x.shape[0] / n_resamples
    resampled = numpy.zeros(n_resamples) * numpy.NaN
    ci = 0.0
    for i in range(resampled.shape[0]):
        si = int(numpy.round(ci))
        ei = int(numpy.round(ci + spb))
        resampled[i] = numpy.nanmean(x[si:ei])
        ci += spb
    
    return resampled


# # # # #
# LOAD DATA

# Choose the correct temporary files.
if DATATYPE == "mouse":
    tmp_dwell_path = DWELLPATH
    tmp_dwell_shape_path = DWELLSHAPEPATH
elif DATATYPE == "gaze":
    tmp_dwell_path = GAZEPATH
    tmp_dwell_shape_path = GAZESHAPEPATH
tmp_resampling_path = RESAMPLEPATH
tmp_resampling_shape_path = RESAMPLESHAPEPATH

# Load the raw data.
if (not os.path.isfile(tmp_dwell_path)) or OVERWRITE_TMP:
    
    print("Loading raw data...")

    # MOUSEVIEW LOAD
    if DATATYPE == "mouse":
        # Load the Gorilla files.
        data = {}
        for fpath in FILEPATHS:
            data_ = gorilla.read_file(fpath, TRIAL_START_ZONE, \
                custom_fields=CUSTOM_FIELDS, use_public_id=True, \
                delimiter=None)
            for key in data_.keys():
                data[key] = copy.deepcopy(data_[key])
            del data_
    
    # GAZE LOAD
    elif DATATYPE == "gaze":
        # Detect all gaze files.
        gaze_files = os.listdir(DATADIR)
        # Loop through all files.
        data = {}
        for fname in gaze_files:
            # Skip files that are not TSV files.
            name, ext = os.path.splitext(fname)
            if ext.lower() != ".tsv":
                continue
            print("\tLoading file {}".format(fname))
            # Save the data under the participant code, without the "subject-"
            # prefix.
            name = name.replace("subject-", "")
            # Add basic info to the current participant.
            data[name] = {}
            data[name]["resolution"] = "1280x1024"
            data[name]["viewport"] = "1280x1024"
            # Load the data.
            fpath = os.path.join(DATADIR, fname)
            data[name]["trials"] = read_opengaze(fpath, missing=0.0)
            for i in range(len(data[name]["trials"])):
                # Convenience renaming of logged messages.
                data[name]["trials"][i]["msg"] = \
                    data[name]["trials"][i]["events"]["msg"]
                # Filter out off-screen data.
                onscreen = (data[name]["trials"][i]["x"] >= 0) & \
                    (data[name]["trials"][i]["x"] <= 1) & \
                    (data[name]["trials"][i]["y"] >= 0) & \
                    (data[name]["trials"][i]["y"] <= 1)
                data[name]["trials"][i]["x"][onscreen==False] = numpy.NaN
                data[name]["trials"][i]["y"][onscreen==False] = numpy.NaN
                # Transform (x,y) from range 0-1 to 0-1280 and 0-1024.
                data[name]["trials"][i]["x"] *= 1280
                data[name]["trials"][i]["y"] *= 1024
    
    # Exclude excluded participants.
    for excluded_name in EXCLUDE:
        if excluded_name in data.keys():
            data.pop(excluded_name)

    # Count the number of participants.
    participants = list(data.keys())
    participants.sort()
    n_participants = len(participants)
    
    # Save the participant IDs.
    pp_memmap = numpy.memmap(PARTICIPANTPATH, dtype="|S32", mode="w+", \
        shape=(len(participants)))
    pp_memmap[:] = participants
    pp_memmap.flush()
    
    print("\nLoaded {} participants".format(n_participants))
    
    # Empty matrices to start with.
    n_conditions = len(CONDITIONS)
    n_stimuli_per_condition = NTRIALS // NTRIALSPERSTIM
    n_aois = len(AOI) - 1
    n_bins = int(numpy.ceil(TRIAL_DURATION / BINWIDTH))
    shape = numpy.array([n_participants, n_conditions, \
        n_stimuli_per_condition, NTRIALSPERSTIM, n_aois, n_bins], \
        dtype=numpy.int32)
    dwell_shape = numpy.memmap(tmp_dwell_shape_path, dtype=numpy.int32, \
        mode="w+", shape=(len(shape)))
    dwell_shape[:] = shape[:]
    dwell_shape.flush()
    dwell = numpy.memmap(tmp_dwell_path, dtype=numpy.float32, mode="w+", \
        shape=tuple(shape))
    dwell[:] = 0.0
    # Empty matrix for resampled data.
    shape = numpy.array([n_participants, 2, n_conditions, \
        n_stimuli_per_condition, NTRIALSPERSTIM, N_RESAMPLES], \
        dtype=numpy.int32)
    resample_shape = numpy.memmap(tmp_resampling_shape_path, \
        dtype=numpy.int32, mode="w+", shape=(len(shape)))
    resample_shape[:] = shape[:]
    resample_shape.flush()
    resamples = numpy.memmap(tmp_resampling_path, dtype=numpy.float32, \
        mode="w+", shape=tuple(shape))
    resamples[:] = 0.0
    
    # Compute the bin edges.
    bin_edges = numpy.arange(0, TRIAL_DURATION+BINWIDTH/2, BINWIDTH, \
        dtype=numpy.float32)
    
    # Run through all participants.
    print("\nLooping through participants...")
    for pi, participant in enumerate(participants):
        
        print("\tProcessing participant '{}' ({}/{})".format( \
            participant, pi+1, len(participants)))

        # Extract the participant's display resolution and view rect.
        dispsize = map(int, data[participant]["resolution"].split("x"))
        viewrect = map(int, data[participant]["viewport"].split("x"))
        
        # Keep track of the count for each affective stimulus.
        stim_count = {}
    
        # Run through all trials.
        for i in range(len(data[participant]["trials"])):
    
            # Start with some blank variables.
            condition = None
            stimuli = {"affective":None, "neutral":None}
            affective_stim = None
            aoi_rect = {"left_stim":None, "right_stim":None}
    
            # Run through all messages in this trial.
            for j, msg in enumerate(data[participant]["trials"][i]["msg"]):
    
                # GAZE
                if DATATYPE == "gaze":
                    # Messages that inform about the gaze files look like the
                    # following: "638571,639272,640306,652333,654352.0,NEUTRAL9.JPG,DISGUST3.JPG,33"
                    if ".JPG" in msg[1]:
                        _, _, _, _, _, right_stim, left_stim, _ = \
                            msg[1].lower().split(",")
                        left_name, ext = os.path.splitext(left_stim)
                        right_name, ext = os.path.splitext(right_stim)
                        for con in CONDITIONS:
                            if (con in left_name) or (con in right_name):
                                condition = con
                        if "neutral" in left_name:
                            affective_stim = "right_stim"
                            stimuli["affective"] = right_name
                            stimuli["neutral"] = left_name
                        elif "neutral" in right_name:
                            affective_stim = "left_stim"
                            stimuli["affective"] = left_name
                            stimuli["neutral"] = right_name
                        aoi_rect["left_stim"] = (80, 332, 480, 360)
                        aoi_rect["right_stim"] = (720, 332, 480, 360)
                        # Stop the message looping; we now have all we need.
                        break
                
                # MOUSEVIEW
                elif DATATYPE == "mouse":
                    # Process the message that contains the condition info.
                    if msg[0] == "condition":
                        condition = msg[1]
        
                    # Process the messages related to the stimulus.
                    elif msg[0] in ["left_stim", "right_stim"]:
                        # Split the extension and the file name.
                        name, ext = os.path.splitext(msg[1])
                        if "neutral" in name:
                            stimuli["neutral"] = name
                        else:
                            stimuli["affective"] = name
                            affective_stim = msg[0]
                    
                    # Process the AOI rect messages.
                    elif msg[0] in ["zone1_shape", "zone2_shape"]:
                        # Extract the rect.
                        rect = msg[1].split("x")
                        rect = map(float, rect)
                        rect = map(round, rect)
                        rect = map(int, rect)
                        if rect[0] < viewrect[0]//2:
                            aoi_rect["left_stim"] = rect
                        else:
                            aoi_rect["right_stim"] = rect
            
            # Skip the practice trial (and trials in other conditions).
            if condition not in CONDITIONS:
                continue
            
            # Skip trials where the stimulus rects don't make sense with the
            # viewport, which should result in only one of the two AOI rects
            # being defined (as both will be lower than half the viewport).
            if (aoi_rect["left_stim"] is None) or \
                (aoi_rect["right_stim"] is None):
                print("\t\tAOI locations for participant " + \
                    "'{}' do not line up with viewrect {}".format( \
                    participant, viewrect))
                break
            
            # Add 1 to the stimulus count.
            if stimuli["affective"] not in stim_count.keys():
                stim_count[stimuli["affective"]] = 0
            else:
                stim_count[stimuli["affective"]] += 1
            
            # In a few rare cases, the task restarted (participant refresh?),
            # and thus stimuli were repeated more than the expected number of
            # times. We simply ignore the excess repetitions.
            if stim_count[stimuli["affective"]] >= NTRIALSPERSTIM:
                continue
            
            # Get the indices for the dwell matrix for the current condition
            # and stimulus.
            i_condition = CONDITIONS.index(condition)
            i_stimulus = STIMULI[condition].index(stimuli["affective"])
            
            if DATATYPE == "gaze":
                # Find the image start and end.
                start_i = 0
                end_i = data[participant]["trials"][i]["time"].shape[0]
                for j, msg in enumerate(data[participant]["trials"][i]["msg"]):
                    if msg[1] == GAZE_START:
                        start_time = msg[0]
                        start_i = numpy.argmin(numpy.abs( \
                            data[participant]["trials"][i]["trackertime"] - \
                            start_time))
                    elif msg[1] == GAZE_IMAGE_END:
                        end_time = msg[0]
                        end_i = numpy.argmin(numpy.abs( \
                            data[participant]["trials"][i]["trackertime"] - \
                            end_time))
                # Select only the data obtained during image viewing.
                for key in ["time", "x", "y", "trackertime", "size"]:
                    data[participant]["trials"][i][key] = \
                        data[participant]["trials"][i][key][start_i:end_i]
                
                # Set the time to 0 at image onset.
                data[participant]["trials"][i]["time"] = \
                    data[participant]["trials"][i]["trackertime"] - \
                    data[participant]["trials"][i]["trackertime"][0]
            
                # Throw out data > 10 seconds of image.
                end_i = numpy.argmin(numpy.abs( \
                    data[participant]["trials"][i]["time"] - 10000))
                for key in ["time", "x", "y", "trackertime", "size"]:
                    data[participant]["trials"][i][key] = \
                        data[participant]["trials"][i][key][:end_i]

            # Resample the (x,y) coordinates.
            x_ = data[participant]["trials"][i]["x"] / float(viewrect[0])
            if affective_stim == "right_stim":
                x_ = 1.0 - x_
            y_ = data[participant]["trials"][i]["y"] / float(viewrect[1])
            t_ = data[participant]["trials"][i]["time"]
            # We need at least a few samples to do this.
            if t_.shape[0] > 10:
                resamples[pi, 0, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], :] = \
                    resample(x_, N_RESAMPLES)
                resamples[pi, 1, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], :] = \
                    resample(y_, N_RESAMPLES)
            else:
                resamples[pi, :, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], :] *= numpy.NaN

            # Go through all samples.
            for si in range(data[participant]["trials"][i]["time"].shape[0]):
                
                # Convenience renaming of the starting time for this sample.
                t0 = data[participant]["trials"][i]["time"][si]
                # Compute the end time of the current sample as the start
                # time of the next.
                if si < data[participant]["trials"][i]["time"].shape[0]-1:
                    t1 = data[participant]["trials"][i]["time"][si+1]
                # If this is the last sample, assume the median inter-sample
                # time.
                else:
                    t1 = t0 + numpy.nanmedian(numpy.diff( \
                        data[participant]["trials"][i]["time"]))
                # Convenience renaming of (x,y) coordinates.
                x = data[participant]["trials"][i]["x"][si]
                y = data[participant]["trials"][i]["y"][si]
                
                # Skip missing data.
                if numpy.isnan(x) or numpy.isnan(y):
                    continue
                # Skip timepoints beyond the edge.
                if t0 >= bin_edges[-1]:
                    continue

                # Check which area of interest this stimulus falls in.
                aoi = "other"
                for aoi_loc in aoi_rect.keys():
                    hor = (x > aoi_rect[aoi_loc][0]) and \
                        (x < aoi_rect[aoi_loc][0]+aoi_rect[aoi_loc][2])
                    ver = (y > aoi_rect[aoi_loc][1]) and \
                        (y < aoi_rect[aoi_loc][1]+aoi_rect[aoi_loc][3])
                    if hor and ver:
                        if aoi_loc == affective_stim:
                            aoi = "affective"
                        else:
                            aoi = "neutral"
                
                # Only record affective and neutral.
                if aoi == "other":
                    continue
                
                # Compute the first bin that this sample falls into. The last
                # bin in the range with smaller edges is the one that the
                # current sample fits within.
                si = numpy.where(bin_edges <= t0)[0][-1]
                
                # Compute the final bin that this sample falls into. The first
                # bin with a larger edge than the current sample will be the
                # bin after the current sample.
                if t1 < bin_edges[-1]:
                    ei = numpy.where(bin_edges > t1)[0][0] - 1
                else:
                    # Minus 2, as we need the last bin, which starts with the
                    # second-to-last bin edge.
                    ei = bin_edges.shape[0] - 2
                
                # If the sample falls within a bin.
                if si == ei:
                    dwell[pi, i_condition, i_stimulus, \
                        stim_count[stimuli["affective"]], AOI.index(aoi), si] \
                        += (t1 - t0) / BINWIDTH
                # If the sample falls in more than one bin.
                else:
                    # Compute the proportion of the first bin that is covered by
                    # the current sample.
                    p0 = (bin_edges[si+1]-t0) / BINWIDTH
                    # Compute the proportion of the last bin that is covered by
                    # the current sample.
                    p1 = (t1 - bin_edges[ei]) / BINWIDTH
                    
                    # Add the proportions to the dwell matrix.
                    dwell[pi, i_condition, i_stimulus, \
                        stim_count[stimuli["affective"]], AOI.index(aoi), si] \
                        += p0
                    dwell[pi, i_condition, i_stimulus, \
                        stim_count[stimuli["affective"]], AOI.index(aoi), ei] \
                        += p1
                    if ei - si > 1:
                        dwell[pi, i_condition, i_stimulus, \
                            stim_count[stimuli["affective"]], AOI.index(aoi), \
                            si+1:ei] = 1.0
    
    # Correct any float errors.
    dwell[dwell>1] = 1.0

# Load the data from the temporary file.
else:
    # Load participant names.
    pp_memmap = numpy.memmap(PARTICIPANTPATH, dtype="|S32", mode="r")
    participants = list(pp_memmap)

    # Load the dwell data's shape from the dwell_shape file.
    dwell_shape = tuple(numpy.memmap(tmp_dwell_shape_path, dtype=numpy.int32, \
        mode="r"))
    # Load the dwell data from file.
    dwell = numpy.memmap(tmp_dwell_path, dtype=numpy.float32, mode="r", \
        shape=dwell_shape)
    # Recompute the bin edges.
    bin_edges = numpy.arange(0, TRIAL_DURATION+BINWIDTH/2, BINWIDTH, \
        dtype=numpy.float32)

# RATINGS
if DO_RATINGS:
    if (not os.path.isfile(RATINGSPATH)) or OVERWRITE_TMP:
        
        # Start with an empty matrix.
        n_participants = len(participants)
        n_conditions = len(CONDITIONS)
        n_stimuli_per_condition = NTRIALS // NTRIALSPERSTIM
        shape = (n_participants, n_conditions, n_stimuli_per_condition)
        ratings_shape = numpy.memmap(RATINGSSHAPEPATH, dtype=numpy.int16, \
            mode="w+", shape=len(shape))
        ratings_shape[:] = numpy.array(shape, dtype=numpy.int16)
        ratings = numpy.memmap(RATINGSPATH, dtype=numpy.float32, mode="w+", \
            shape=shape)
        ratings[:] = numpy.NaN
        
        # Open the ratings CSV.
        raw_ratings = {}
        raw = numpy.loadtxt(RATINGFILEPATH, dtype=str, unpack=True, delimiter=",")
        for i in range(raw.shape[0]):
            var = raw[i,0]
            val = raw[i,1:]
            if var in ["subject"]:
                raw_ratings[var] = val
            else:
                try:
                    raw_ratings[var] = val.astype(numpy.float32)
                except:
                    raw_ratings[var] = val
        
        # Go though all participants.
        for ppi, ppcode in enumerate(participants):
            # Find the participant in the ratings data.
            i = numpy.where(raw_ratings["subject"]==ppcode)[0]
            if i.shape[0] > 0:
                i = i[0]
            else:
                continue
            # Go through all conditions.
            for ci, condition in enumerate(CONDITIONS):
                # Go through all stimuli.
                for si in range(ratings.shape[2]):
                    # Construct the stimulus name for mouse or gaze data.
                    if condition == "disgust":
                        varname = "{}{}.disgusted.rate".format(condition, si+1)
                    elif condition == "pleasant":
                        varname = "{}{}.pleasant or unpleasant.rate".format(condition, si+1)
                    else:
                        varname = "{}{}.{}.rate".format(condition, si+1, \
                            condition)
                    ratings[ppi,ci,si] = raw_ratings[varname][i]
        ratings.flush()
    
    else:
        # Load rating data.
        ratings_shape = tuple(numpy.memmap(RATINGSSHAPEPATH, dtype=numpy.int16, \
            mode="r"))
        ratings = numpy.memmap(RATINGSPATH, dtype=numpy.float32, mode="r", \
            shape=ratings_shape)
    
    # Recode the data into long format.
    with open(os.path.join(TMPDATADIR, "{}_long.csv".format(DATATYPE)), "w") as f:
        # Construct and write the header.
        header = ["subject", "method", "condition", "stimulus_nr", \
            "presentation_nr", "stimulus_rating", "dwell"]
        f.write(",".join(header))
        # Loop through all participants, conditions, stimuli, and 
        # presentations.
        for ppi in range(dwell.shape[0]):
            for ci in range(dwell.shape[1]):
                condition = CONDITIONS[ci]
                for si in range(dwell.shape[2]):
                    rating = ratings[ppi,ci,si]
                    for pi in range(dwell.shape[3]):
                        d = numpy.nanmean(dwell[ppi,ci,si,pi,0,:] - \
                            dwell[ppi,ci,si,pi,1,:])
                        line = [ppi,DATATYPE, condition, si+1, pi+1, rating, d]
                        f.write("\n" + ",".join(map(str, line)))


# # # # #
# RELIABILITY

# R and standard error for the Spearman-Brown split-half reliability.
sb_splithalf = { \
    "r":{}, \
    "se":{}, \
    }
# Cronbach alpha will be a single measure per condition.
coeff_alpha = {}

# Loop through the conditions.
for ci, condition in enumerate(CONDITIONS):
    
    # Compute the difference between neutral and affective stimulus. This
    # quantifies approach (positive values) or avoidance (negative values).
    # Computed as affective minus neutral, only for the current condition.
    # d has shape (n_participants, n_stimuli, n_presentations, n_bins)
    d = dwell[:,ci,:,:,0,:] - dwell[:,ci,:,:,1,:]
    
    # Average the dwell difference across all time bins.
    # d_m has shape (n_participants, n_stimuli, n_presentations)
    d_m = numpy.nanmean(d, axis=3)
    
    # Compute split-half reliability across all trials, regardless of stimulus
    # and presentation number. In order to do so, we first collapse all
    # stimuli and presentations together.
    d_sb = numpy.zeros((d_m.shape[1]*d_m.shape[2], d_m.shape[0]), \
        dtype=d_m.dtype)
    for j in range(d_m.shape[1]):
        for k in range(d_m.shape[2]):
            d_sb[j*k+k,:] = d_m[:,j,k]
    sb_splithalf["r"][condition], sb_splithalf["se"][condition] = \
        split_half(d_sb, n_splits=100, mode="spearman-brown")
    
    # Compute Cronbach's alpha on the same stimuli.
    coeff_alpha[condition] = cronbach_alpha(d_sb)

# Write the split-half reliability to file.
fpath = os.path.join(OUTDIR, "{}_splithalf.csv".format(DATATYPE))
with open(fpath, "w") as f:
    header = ["condition", "R", "se", "alpha"]
    f.write(",".join(map(str, header)))
    for condition in sb_splithalf["r"].keys():
        line = [condition, sb_splithalf["r"][condition], \
            sb_splithalf["se"][condition], coeff_alpha[condition]]
        f.write("\n" + ",".join(map(str, line)))


# # # # #
# PLOTTING

# DWELL PER AOI
# Create a two-panel figure. (Columns for different conditions.)
fig, axes = pyplot.subplots(nrows=1, ncols=len(CONDITIONS), \
    figsize=(8.0*len(CONDITIONS), 6.0), dpi=300.0)
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.94,
    wspace=0.15, hspace=0.2)

# Compute the bin centres to serve as x-values.
time_bin_centres = bin_edges[:-1] + numpy.diff(bin_edges)/2.0

# Loop through the conditions.
for ci, condition in enumerate(CONDITIONS):
    
    # Choose the top-row, and the column for this condition.
    ax = axes[ci]
    
    # Loop through all AOIs.
    for ai in range(dwell.shape[4] + 1):

        # Grab the dwell proportion for this AOI, or for "other".
        # d has shape (n_participants, n_stimuli, n_presentations, n_bins)
        if ai < dwell.shape[4]:
            d = dwell[:,ci,:,:,ai,:]
        else:
            d = 1.0 - numpy.nansum(dwell[:,ci,:,:,:,:], axis=3)
        
        # Average over all different stimuli, and recode as percentages.
        # val has shape (n_participants, n_presentations, n_bins)
        val = 100 * numpy.nanmean(d, axis=1)
        
        # Compute the mean over all participants.
        m = numpy.nanmean(val, axis=0)
        # Compute the within-participant 95% confidence interval.
        nv = val - numpy.nanmean(val, axis=0) \
            + numpy.nanmean(numpy.nanmean(val, axis=0))
        sd = numpy.nanstd(nv, axis=0, ddof=1)
        sem = sd / numpy.sqrt(nv.shape[0])
        ci_95 = 1.96 * sem
        
        # Compute where the averages are signiticantly different from 0.
        t_val, p_val = ttest_1samp(val, 0.0)
        # Compute effect size (Cohen's d_z).
        sig = numpy.ones(p_val.shape, dtype=bool)
        # Correct for multiple comparisons.
        if MULTIPLE_COMPARISONS_CORRECTION is None:
            sig[p_val > 0.05] = 0
        elif MULTIPLE_COMPARISONS_CORRECTION is "no correction":
            pass
        elif type(MULTIPLE_COMPARISONS_CORRECTION) == float:
            sig[p_val > MULTIPLE_COMPARISONS_CORRECTION] = 0
        elif MULTIPLE_COMPARISONS_CORRECTION == "bonferroni":
            sig[p_val > 0.05/p_val.shape[1]] = 0.0
        elif MULTIPLE_COMPARISONS_CORRECTION == "holm":
            for j in range(p_val.shape[0]):
                sorted_i = numpy.argsort(p_val[j,:])
                alpha = 0.05 / numpy.arange(p_val.shape[1],0.9,-1)
                cutoff = numpy.where(p_val[j,:][sorted_i] > alpha)[0][0]
                sig[j,sorted_i[cutoff:]] = 0
    
        # Specify the colour map for the current condition.
        if ai == 0:
            cmap = PLOTCOLMAPS[condition]
        elif ai == 1:
            cmap = "Greens"
        else:
            cmap = "Greys"
        cmap = matplotlib.cm.get_cmap(cmap)
        voffset = 3
        vmin = 0
        vmax = dwell.shape[3] + voffset
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
        # LINES
        # Plot each stimulus presentation separately.
        for j in range(m.shape[0]):
            
            # Define the label.
            if j == 0:
                lbl = "First presentation"
            else:
                lbl = "Stimulus repeat {}".format(j)
            if ai < dwell.shape[4]:
                lbl = None
    
            # PLOT THE LIIIIIIINE!
            ax.plot(time_bin_centres, m[j,:], "-", color=cmap(norm(j+voffset)), \
                lw=3, label=lbl)
            # Shade the confidence interval.
            ax.fill_between(time_bin_centres, m[j,:]-ci_95[j,:], \
                m[j,:]+ci_95[j,:], alpha=0.2, color=cmap(norm(j+voffset)))
        
    # Add a legend.
    ax.legend(loc="upper right", fontsize=FONTSIZE["legend"])

    # Finish the plot.
    ax.set_title(condition.capitalize(), fontsize=FONTSIZE["axtitle"])
    ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
    ax.set_xlim([0, TRIAL_DURATION])
    ax.set_xticks(range(0, TRIAL_DURATION+1, 1000))
    ax.set_xticklabels(range(0, (TRIAL_DURATION//1000 + 1)), \
        fontsize=FONTSIZE["ticklabels"])
    if ci == 0:
        ax.set_ylabel(r"$\Delta$dwell time ({} \% pt.)".format(DATATYPE), \
            fontsize=FONTSIZE["label"])
    ax.set_ylim([0, 100])
    ax.set_yticks(range(0, 100+1, 10))
    ax.set_yticklabels(range(0, 100+1, 10), fontsize=FONTSIZE["ticklabels"])

# Save the figure.
fig.savefig(os.path.join(OUTDIR, "dwell_per_AOI_{}.png".format(DATATYPE)))
pyplot.close(fig)


# DWELL TIME DIFFERENCES
# The dwell matrix has shape (n_participants, n_conditions, 
# n_stimuli_per_condition, n_stimulus_presentations, n_aois, n_bins)
# Plotting should happen as a time series (dwell[:,:,:,:,:,0:t). We're 
# interested in the difference between the neutral AOI  (dwell[:,:,:,:,1,:])
#  and the affective AOI (dwell[:,:,:,:,0,:]). This should be averaged across
# all stimuli (dwell[:,:,0:n,:,:,:]), and plotted separately for each stimulus
# presentation (dwell[:,:,:,0:NTRIALSPERSTIM,:,:]). The conditions 
# (dwell[:,i,:,:,:,:]) should be plotted in separate plots. The average dwell
# time difference and 95% confidence interval should be computed over all
# participants (dwell[0:n_participants,:,:,:,:,:]).

# Create a four-panel figure. (Top row for lines, bottom row for heatmaps;
# columns for different conditions.)
fig, axes = pyplot.subplots(nrows=2, ncols=len(CONDITIONS), \
    figsize=(8.0*len(CONDITIONS),9.0), dpi=300.0)
fig.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.95,
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
    d = dwell[:,ci,:,:,0,:] - dwell[:,ci,:,:,1,:]
    
    # Difference after averaging all stimuli.
    dwell_stim = numpy.nanmean(dwell, axis=2)
    
    # Average over all different stimuli, and recode as percentages.
    # val has shape (n_participants, n_presentations, n_bins)
    val = 100 * numpy.nanmean(d, axis=1)
    
    # Compute the mean over all participants.
    m = numpy.nanmean(val, axis=0)
    # Compute the within-participant 95% confidence interval.
    nv = val - numpy.nanmean(val, axis=0) \
        + numpy.nanmean(numpy.nanmean(val, axis=0))
    sd = numpy.nanstd(nv, axis=0, ddof=1)
    sem = sd / numpy.sqrt(nv.shape[0])
    ci_95 = 1.96 * sem
    
    # Compute where the averages are signiticantly different from 0.
    t_val, p_val = ttest_1samp(val, 0.0)
    # Compute effect size (Cohen's d_z).
    sig = numpy.ones(p_val.shape, dtype=bool)
    # Correct for multiple comparisons.
    if MULTIPLE_COMPARISONS_CORRECTION is None:
        sig[p_val > 0.05] = 0
    elif MULTIPLE_COMPARISONS_CORRECTION is "no correction":
        pass
    elif type(MULTIPLE_COMPARISONS_CORRECTION) == float:
        sig[p_val > MULTIPLE_COMPARISONS_CORRECTION] = 0
    elif MULTIPLE_COMPARISONS_CORRECTION == "bonferroni":
        sig[p_val > 0.05/p_val.shape[1]] = 0.0
    elif MULTIPLE_COMPARISONS_CORRECTION == "holm":
        for j in range(p_val.shape[0]):
            sorted_i = numpy.argsort(p_val[j,:])
            alpha = 0.05 / numpy.arange(p_val.shape[1],0.9,-1)
            cutoff = numpy.where(p_val[j,:][sorted_i] > alpha)[0][0]
            sig[j,sorted_i[cutoff:]] = 0

    # Specify the colour map for the current condition.
    cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
    voffset = 3
    vmin = 0
    vmax = dwell.shape[3] + voffset
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the separation line and background colours.
    ax.plot(time_bin_centres, numpy.zeros(time_bin_centres.shape), ':', lw=3, \
        color="black", alpha=0.5)
    ax.fill_between(time_bin_centres, \
        numpy.ones(time_bin_centres.shape)*YLIM["dwell_p"][0], \
        numpy.zeros(time_bin_centres.shape), color="black", alpha=0.05)
    annotate_x = time_bin_centres[0] + \
        0.02*(time_bin_centres[-1]-time_bin_centres[0])
    ax.annotate("Approach", (annotate_x, YLIM["dwell_p"][1]-7), \
        fontsize=FONTSIZE["annotation"])
    ax.annotate("Avoidance", (annotate_x, YLIM["dwell_p"][0]+3), \
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
        ax.plot(time_bin_centres, m[j,:], "-", color=cmap(norm(j+voffset)), \
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
        ax.set_ylabel(r"$\Delta$dwell time ({} \% pt.)".format(DATATYPE), \
            fontsize=FONTSIZE["label"])
    ax.set_ylim(YLIM["dwell_p"])
    ax.set_yticks(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10))
    ax.set_yticklabels(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10), \
        fontsize=FONTSIZE["ticklabels"])
    
    # HEATMAP
    # Choose the heatmap row, and the column for the current condition.
    ax = axes[1,ci]
    
    heatmap = numpy.copy(t_val)
    heatmap[sig==False] = numpy.NaN
    plotcol_name = "t_vals"
    bar_lbl = r"$t$ statistic"

    # Create the colourmap for this condition.
    cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[plotcol_name])
    vmin = YLIM[plotcol_name][0]
    vmax = YLIM[plotcol_name][1]
    vstep = numpy.min(numpy.abs(YLIM[plotcol_name])) // 2

    # Plot the heatmap.
    ax.imshow(heatmap, cmap=cmap, vmin=vmin, vmax=vmax, \
        interpolation="none", aspect="auto", origin="upper")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    bax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
        ticks=range(vmin, vmax+1, vstep), orientation='vertical')
    if ci == axes.shape[1]-1:
        cbar.set_label(bar_lbl, fontsize=FONTSIZE["bar"])
    cbar.ax.tick_params(labelsize=FONTSIZE["ticklabels"])

    ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
    ax.set_xlim([-0.5, heatmap.shape[1]-0.5])
    ax.set_xticks(numpy.linspace(-0.5, heatmap.shape[1]+0.5, \
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
fig.savefig(os.path.join(OUTDIR, "dwell_percentages_{}.png".format(DATATYPE)))
pyplot.close(fig)


# RATINGS
if DO_RATINGS:
    # Plot the ratings for all stimuli in each condition. Each stimulus gets its
    # own panel, and each condition its own row of panels. Repeated presentations
    # of the same stimuli are plotted in the same window.
    n_participants, n_conditions, n_stimuli = ratings.shape
    fig, ax = pyplot.subplots(nrows=n_conditions, ncols=n_stimuli, \
        figsize=(8.0*n_stimuli,6.0*n_conditions), dpi=100.0)
    fig.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.95,
        wspace=0.15, hspace=0.2)
    
    # Run through all conditions.
    for ci in range(n_conditions):
    
        # Grab the condition name.
        condition = CONDITIONS[ci]
        
        # Specify the colour map for the current condition.
        cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
        voffset = 3
        vmin = 0
        vmax = dwell.shape[3] + voffset
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
        # Loop through all stimuli.
        for si in range(n_stimuli):
            
            # Compute the dwell difference, averaged over time bins.
            d = numpy.nanmean(dwell[:,ci,si,:,0,:] - dwell[:,ci,si,:,1,:], axis=2)
            # Transform to percentages.
            d *= 100
            
            # Run through each stimulus repetition.
            for ri in range(d.shape[1]):
                
                # Determine the colour for this repeat.
                col = cmap(norm(ri+voffset))
    
                # Find the order of all x-values, for plotting purposes.
                # NOTE: x will be used to order, and y values will be ordered
                # according to x's order. Hence, the correct values will remain
                # linked. Separate ordering would be exceptionally wrong.
                notnan = (numpy.isnan(ratings[:,ci,si])==False) & \
                    (numpy.isnan(d[:,ri])==False)
                order = numpy.argsort(ratings[:,ci,si][notnan])
                x = statsmodels.api.add_constant(ratings[:,ci,si][notnan][order])
                y = d[:,ri][notnan][order]
    
                # Plot the individual samples.
                ax[ci,si].plot(x[:,1], y, "o", color=col, alpha=0.3)
                
                # Run the regression.
                model = statsmodels.api.OLS(y, x)
                result = model.fit()
                # Get the fit details.
                st, dat, ss2 = summary_table(result, alpha=0.05)
                y_pred = dat[:,2]
                predict_mean_se  = dat[:,3]
                predict_mean_ci_low, predict_mean_ci_upp = dat[:,4:6].T
                predict_ci_low, predict_ci_upp = dat[:,6:8].T
    
                # Plot the regression line.
                ax[ci,si].plot(x[:,1], y_pred, "-", lw=3, color=col, alpha=0.8)
                # Plot regression confidence envelope.
                ax[ci,si].fill_between(x[:,1], predict_mean_ci_low, \
                    predict_mean_ci_upp, color=col, alpha=0.2)
                # Plot prediction intervals.
                #ax[ci,si].plot(x[:,1], predict_ci_low, '-', lw=2, color=col, alpha=0.5)
                #ax[ci,si].plot(x[:,1], predict_ci_upp, '-', lw=2, color=col, alpha=0.5)
                
                # Set plot title.
                ax[ci,si].set_title(STIMULI[condition][si], fontsize=FONTSIZE["label"])
                
                # Set limits.
                ax[ci,si].set_xlim([0,100])
                ax[ci,si].set_ylim([-100,100])
                
    
        ax[ci,0].set_ylabel(r"$\Delta$dwell time ({} \% pt.)".format(DATATYPE), \
            fontsize=FONTSIZE["label"])
        ax[ci,2].set_xlabel("Stimulus {} rating".format(condition), \
            fontsize=FONTSIZE["label"])
    
    fig.savefig(os.path.join(OUTDIR, "ratings_dwell_correlation_{}.png".format( \
        DATATYPE)))
    pyplot.close(fig)

