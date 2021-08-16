import os
import copy

import numpy

from pygazeanalyser.detectors import blink_detection, fixation_detection, \
    saccade_detection

def read_opengaze(fpath, missing=0.0):
    
    # Read the content.
    with open(fpath, "r") as f:
        lines = f.readlines()

    # Grab the header.
    header = lines.pop(0)
    header = header.replace('\n','').replace('\r','').split('\t')
    
    # Empty list to store all trials in.
    data = []

    # Loop through all data.
    prev_m = "0"
    new_trial = True
    for i, line in enumerate(lines):
        
        # Start new lists.
        if new_trial:
            x = []
            y = []
            size = []
            trackertime = []
            events = {'msg':[]}
            # Toggle the new_trial indicator.
            new_trial = False
        
        # Reformat the line into a list.
        line = line.replace('\n','').replace('\r','').split('\t')

        # Add all values to their respective lists.
        # All lines (when obtained through PyOpenGaze or PyGaze)
        # should contain the following data:
        #     CNT, TIME, TIME_TICK, 
        #     FPOGX, FPOGY, FPOGS, FPOGD, FPOGID, FPOGV,
        #     LPOGX, LPOGY, LPOGV, RPOGX, RPOGY, RPOGV,
        #     BPOGX, BPOGY, BPOGV,
        #     LPCX, LPCY, LPD, LPS, LPV, RPCX, RPCY, RPD, RPS, RPV
        #     LEYEX, LEYEY, LEYEZ, LPUPILD, LPUPILV,
        #     REYEX, REYEY, REYEZ, RPUPILD, RPUPILV,
        #     CX, CY, CS, USER
        try:
            # Compute the size of the pupil.
            left = line[header.index("LPV")] == '1'
            right = line[header.index("RPV")] == '1'
            if left and right:
                s = (float(line[header.index("LPD")]) + \
                    float(line[header.index("RPD")])) / 2.0
            elif left and not right:
                s = float(line[header.index("LPD")])
            elif not left and right:
                s = float(line[header.index("RPD")])
            else:
                s = 0.0
            # Extract data.
            x.append(float(line[header.index("BPOGX")]))
            y.append(float(line[header.index("BPOGY")]))
            size.append(s)
            trackertime.append(int(1000 * float(line[header.index("TIME")])))

        # Skip lines that cannot be parsed.
        except:
            print("line '%s' could not be parsed" % line)
            continue
        
        # Read the USER column. If it is not "0", then it is a user-defined
        # message.
        if line[header.index("USER")] != '0':
            # Grab the message time and content.
            t = int(1000 * float(line[header.index("TIME")]))
            m = line[header.index("USER")]
            
            # Only act on non-double messages.
            if m != prev_m:
                # Check if this message signifies the end of the current trial.
                if ".JPG" in m:
                    # Parse the message. It will be in the following form:
                    # [recording_onset, fixation_onset, image_onset, ITI_onset, \
                    #    endmessage_time, right_image, left_image, trialnr]
                    trial_start, fix_onset, img_onset, iti_onset, msg_time, \
                        right_img, left_img, trialnr = m.split(",")
                    # Round message time.
                    msg_time = int(round(float(msg_time)))
    
                    # Use the time differences to reconstruct the events.
                    offset = t - msg_time
                    events['msg'].append([int(trial_start) + offset, "START_TRIAL"])
                    events['msg'].append([int(fix_onset)   + offset, "FIX_START"])
                    events['msg'].append([int(img_onset)   + offset, "IMAGE_START"])
                    events['msg'].append([int(iti_onset)   + offset, "ITI_START"])
                    events['msg'].append([msg_time         + offset, m])
                    events['msg'].append([t, m])
                    events['msg'].append([msg_time         + offset, "STOP_TRIAL"])
                    
                    # Toggle the new_trial indicator.
                    new_trial = True

            # Save the current message as the previous.
            prev_m = copy.copy(m)
        
        # Save this trial's data.
        if new_trial:
            trial = {}
            trial['x'] = numpy.array(x)
            trial['y'] = numpy.array(y)
            trial['size'] = numpy.array(size)
            trial['trackertime'] = numpy.array(trackertime)
            trial['time'] = trial['trackertime'] - trial['trackertime'][0]
            trial['events'] = copy.deepcopy(events)
            trial['events']['Sblk'], trial['events']['Eblk'] = \
                blink_detection(trial['x'], trial['y'], \
                trial['trackertime'], missing=missing)
            trial['events']['Sfix'], trial['events']['Efix'] = \
                fixation_detection(trial['x'], trial['y'], \
                trial['trackertime'], missing=missing)
            trial['events']['Ssac'], trial['events']['Esac'] = \
                saccade_detection(trial['x'], trial['y'], \
                trial['trackertime'], missing=missing)
            data.append(trial)

    return data

    