"""
HOTSPOTTING with BRAINVISION
----------------------------

Author: Angus McMorland, a.mcmorland@auckland.ac.nz
Date: 2017-05-03

Dependencies
------------
python vers 2.x
numpy, matplotlib - available separately or (recommended) from Anaconda Python 
                    distribution
PySide - installation instructions are here:
         http://pyside.readthedocs.io/en/latest/installing/windows.html
       - could probably be adapted to use pyqt (also in Anaconda)
         get in touch if you need help to do this, or if you add this feature,
         please submit back to the github repo
libRDA - from BrainVision

Usage
-----
Variables are defined under "# config options" near the top of the file.

Call on the command line as: "python hotspot_brainvision.py"

Notes
-----
BrainVision Recorder RDA stats:
This program receives a data stream from BrainVision Recorder via a TCP/IP 
socket, referred to as remote data access (RDA). This interface is detailed
in chapter 11 of the BrainVision Recorder 1.20 User Manual (vers 007, Mar 2012).  

The program graphically displays a sweep of data from a specified channel, 
defined as "channel_of_interest".  In this context "channel_of_interest" is 
the EEG channel being monitored.  
The sampling rate in hertz of the incoming data is defined by "sampling_freq". 
The number of data points stored for each sweep is defined by "nstore".  
Data for the period in seconds from "pretrig_time" to "posttrig_time" is 
displayed on a graph, with time on the x-axis.  The scale of the y-axis is 
defined in microvolts by "y_scale".   
Other data are displayed in a text window.  The RMS is calculated and 
displayed for the period in seconds from "bg_window_start" to "bg_window_stop".  
The difference between the maximum and minimum voltage (peak-to-peak amplitude 
of the MEP) in the period defined from "mep_window_start" to "mep_window_stop" 
is also displayed.  

The marker around which a sweep is defined is specified in the line of code 
"if m.markers[i].description ==".  NOTE: this is defined by four characters 
that include spaces, e.g., an S128 marker is defined as 'S128', an S16 marker 
is defined as "S 16", an S4 marker is defined as "S  4". Multiple triggers for 
a sweep can be defined using the code "if m.markers[i].description in 
[...]". For example, if markers S1, S2 and S3 identify TMS stimuli following 
which a sweep is to be displayed, then these can be coded 
"if m.markers[i].description in ['S  1', 'S  2', 'S  3']".

50 Hz packets (20 ms b/w)
each packet has all channels of 100 pts

Licence
-------
BSD 3-Clause License (Revised)
Copyright (c) 2017, Angus McMorland, University of Auckland
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Auckland nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ANGUS MCMORLAND BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import matplotlib
matplotlib.rcParams['backend.qt4'] = 'PySide'
matplotlib.use('Qt4Agg')

import numpy as np

from PySide import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from libRDA import RDAConnection, GetRDAMessage

# --------------------------------------------------------------------
# config options
sampling_freq       = 5000    # Hz
nstore              = 10000   # samples to store
pretrig_time        = 100e-3  # s
posttrig_time       = 300e-3  # s

channel_of_interest = 32      # leads start at 0, not 1

bg_window_start     = -55e-3  # s relative to trig
bg_window_stop      = -5e-3   # ditto
mep_window_start    = 20e-3   # ''
mep_window_stop     = 45e-3   # ''
nsd                 = 3       # number of std devs above RMS for threshold

bg_rms_threshold    = 50      # microvolts
y_scale             = 1000    # microvolts
# --------------------------------------------------------------------

def rms(arr):
    return np.sqrt((arr**2).mean())

def t2s(t):
    return int(t * sampling_freq)
    
def s2t(s):
    return s / float(sampling_freq)

# calc'd options
pretrig_samples  = t2s(pretrig_time)
posttrig_samples = t2s(posttrig_time)

class FIFOArray1D(object):
    def __init__(self, shape):
        self.data = np.empty(shape)  # official location
        self.other = np.empty(shape) # explicit temp space
        
    def add(self, new_data):
        len_new = len(new_data)
        len_me = self.data.size
        if len_new <= len_me:
            self.other[-len_new:] = new_data
            self.other[:-len_new] = self.data[len_new:]
            self.data[:] = self.other[:]
        else:
            self.data[:] = new_data[-len_me:]
        
    def get_data(self):
        return self.data

class HotSpotter(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.init_gui()
        self.rda = RDAConnection()
        
        self.sanity_check_times()
        
        # init vars
        self.finish = False
        
        self.data = FIFOArray1D((nstore,))
        self.sample = np.empty((pretrig_samples + posttrig_samples,))
        self.trigat = None
        
        # run code
        timer = QtCore.QTimer(self)
        QtCore.QObject.connect(timer, QtCore.SIGNAL("timeout()"), \
                               self.timer_event)
        timer.start(5)

    def init_gui(self):
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("SpikePlotter")
        self.setGeometry(500,160,960,700)
        
        fig = Figure(figsize=(1000,1000), dpi=72, facecolor=(1,1,1), \
                     edgecolor=(0,0,0))
        ax = fig.add_subplot(111)
        self.line = ax.plot([0,1] ,'-')[0]
        ax.set_ylim(-y_scale, y_scale)
        ax.set_xlim(-pretrig_time, posttrig_time)
        
        # draw trig
        ax.axvline(0, color='r')
        
        # draw window regions - will need to make this updateable later
        ax.axvspan(bg_window_start, bg_window_stop, facecolor='b', alpha=0.3)
        ax.axvspan(mep_window_start, mep_window_stop, facecolor='r', alpha=0.3)
        
        # generate the canvas to display the plot
        canvas = FigureCanvas(fig)
        self.setCentralWidget(canvas)

    def sanity_check_times(self):
        # check that -pretrig_time < bg_window_start
        if -pretrig_time > bg_window_start:
            print "Pretrig time needs to be greater than background window start."
            return -1
        # check that posttrig_time > mep_window_stop
        if posttrig_time < mep_window_stop:
            print "Posttrig time needs be greater than MEP window stop."
            return -1
        return 0

    def calc_bg_stats(self, data):
        '''
        data : ndarray
          1-D array of data samples from pretrig to posttrig
        '''
        # isolate pretrig window
        start = pretrig_samples + t2s(bg_window_start) 
            # since bg_window_start is -ve
        stop  = pretrig_samples + t2s(bg_window_stop)
        bg_window = data[start:stop] - np.mean(data[start:stop]) 
            #Baseline correct bg_window
        
        return rms(bg_window), bg_window.std()

    def calc_mep_stats(self, data, bg_rms, bg_sd):
        '''
        data : ndarray
          1-D array of data samples from pretrig to posttrig
        '''
        # isolate posttrig window
        threshold = bg_rms + nsd * bg_sd
        start = pretrig_samples + t2s(mep_window_start)
        stop  = pretrig_samples + t2s(mep_window_stop)
        #print "MEP window from %d to %d (%d)" % (start, stop, stop-start)
        mep_window = data[start:stop]
        
        above = mep_window > threshold
        if np.any(above):
            first_idx = np.flatnonzero(above)[0]
            latency = s2t(first_idx)
        else:
            latency = None
            
        # peak-to-peak, simply max - min
        ptp = mep_window.max() - mep_window.min()
            
        return latency, ptp
        
    def find_amplitude(self, mep_window):
        max = np.amax(mep_window)
        min = np.amin(mep_window)
        amplitude = max - min
        return amplitude 

    def check_mep_criteria(self, bg_rms, latency):
        if (bg_rms > bg_rms_threshold):
            return False
        if (latency == None):
            return False
        return True

    def update_plot(self, data):
        ax = self.line.axes
        self.line.set_data(np.linspace(-pretrig_time, posttrig_time, \
                           len(data)), data - data[0])
        #ax.relim()
        self.line.figure.canvas.draw()

    def timer_event(self):
        m = GetRDAMessage(self.rda)
        
        if m.msgtype == 1:
            print "Start"
            print "Number of channels: " + str(self.rda.channelCount)
            print "Sampling interval: " + str(self.rda.samplingInterval) 
                # microseconds
            print "Resolutions: " + str(self.rda.resolutions)
            print "Channel Names: "
            
            print "\n".join((['%02d : %s' % (i, x) for i, x in \
                enumerate(self.rda.channelNames)]))
            
            print "Channel of interest: %s" % \
                (self.rda.channelNames[channel_of_interest])
            print "Resolution: %0.3f" % \
                (self.rda.resolutions[channel_of_interest])

        elif m.msgtype == 4:
            nch = self.rda.channelCount
        
            # Put data at the end of actual buffer
            # isolate channel of interest
            ch_data = np.array(m.data[channel_of_interest::nch])
            ch_data *= self.rda.resolutions[channel_of_interest]

            # add to FIFO
            self.data.add(ch_data)
            
            # ... and shift any running trigs
            if self.trigat != None:
                self.trigat += len(ch_data)

            # Print markers, if there are some in actual block
            if m.markerCount > 0:
                for i in range(m.markerCount):
                    print "Marker " + m.markers[i].description + \
                        " of type " + m.markers[i].type + \
                        " at position " + str(m.markers[i].position) + \
                        " with points " + str(m.markers[i].points)
                        
                    # if marker descr == correct S marker then display sweep
                    if m.markers[i].description == 'S128':
                        print "*"
                        self.trigat  = int(m.points - m.markers[i].position)
                        # this will get last one if >1 S128
                        
            if self.trigat > posttrig_samples:
                # have collected enough data to display
                
                start = -self.trigat - pretrig_samples
                stop  = -self.trigat + posttrig_samples
                
                print start, stop, stop-start # should be 
                s = slice(start, stop)
                print "Data shape", self.data.get_data().shape
                window = self.data.get_data()[s]
                print "Window shape", window.shape
                self.update_plot(window)
                
                bg_rms, bg_sd = self.calc_bg_stats(window)
                print "Background RMS: %0.4f" % (bg_rms)
                latency, ptp = self.calc_mep_stats(window, bg_rms, bg_sd)
                if latency != None:
                    print "Latency: %0.4f s" % (latency)
                else:
                    print "No supra-threshold MEP detected."
                print "Peak-to-peak: %0.4f" % (ptp)
                print "Good MEP?", self.check_mep_criteria(bg_rms, latency)
                self.trigat = None
                
        elif m.msgtype == 3:
            self.finish = True

app = QtGui.QApplication(sys.argv)
frame = HotSpotter()
frame.show()

exit_status = app.exec_()
frame.rda.close()
sys.exit(exit_status)
