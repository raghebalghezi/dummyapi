import math
import parselmouth
import statistics
from parselmouth.praat import call
import numpy as np

def speech_rate(filename):
    silencedb = -25
    mindip = 2
    minpause = 0.3
    sound = parselmouth.Sound(filename)
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0.001
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    asd = speakingtot / (voicedcount + 0.0001)
    speechrate_dictionary = {'nsyll':voicedcount,
                             'npause': npause,
                             'dur(s)':originaldur,
                             'phonationtime(s)':intensity_duration,
                             'speechrate(nsyll / dur)': speakingrate,
                             "articulation rate(nsyll / phonationtime)":articulationrate,
                             "ASD(speakingtime / nsyll)":asd}

    return speechrate_dictionary


def get_formant_attributes(filename, time_step=0., pitch_floor=75., pitch_ceiling=600.,
                           max_num_formants=5., max_formant=5500.,
                           window_length=0.025, pre_emphasis_from=50.,
                           unit='Hertz', interpolation_method='Linear', replacement_for_nan=0.):
    """
    Function to get formant-related attributes such as mean and median formants.
    Adapted from David Feinberg's work: https://github.com/drfeinberg/PraatScripts

    :param (parselmouth.Sound) sound: sound waveform
    :param (float) time_step: the measurement interval (frame duration), in seconds (default: 0.0)
    :param (float) pitch_floor: minimum pitch (default: 75.)
    :param (float) pitch_ceiling: maximum pitch (default: 600.)
    :param (float) max_num_formants: maximum number of formants for analysis (default: 5.)
    :param (float) max_formant: maximum allowed frequency for a formant (default: 5500.)
           NOTE: The default value of 5500. corresponds to an adult female.
    :param (float) window_length: the duration of the analysis window, in seconds (default: 0.025)
    :param (float) pre_emphasis_from: the frequency F above which the spectral slope will
           increase by 6 dB/octave (default: 50.)
    :param (str) unit: units of the result, 'Hertz' or 'Bark' (default: 'Hertz)
    :param (str) interpolation_method: method of sampling new data points with (default: 'Linear)
    :param (float) replacement_for_nan: a float number that will represent frames with NaN values
           (default: 0.)
    :return: a dictionary of mentioned attributes
    """
    sound = parselmouth.Sound(filename)

    # Create PointProcess object
    point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)

    # Create Formant object
    formant = call(sound, "To Formant (burg)", time_step, max_num_formants, max_formant,
                   window_length, pre_emphasis_from)

    # Get number of points in PointProcess
    num_points = call(point_process, "Get number of points")
    if num_points == 0:
        return dict(), None

    f1_list, f2_list, f3_list, f4_list = [], [], [], []

    # Measure formants only at glottal pulses
    for point in range(1, num_points+1):
        t = call(point_process, "Get time from index", point)
        f1 = call(formant, "Get value at time", 1, t, unit, interpolation_method)
        f2 = call(formant, "Get value at time", 2, t, unit, interpolation_method)
        f3 = call(formant, "Get value at time", 3, t, unit, interpolation_method)
        f4 = call(formant, "Get value at time", 4, t, unit, interpolation_method)
        f1_list.append(f1 if not math.isnan(f1) else replacement_for_nan)
        f2_list.append(f2 if not math.isnan(f2) else replacement_for_nan)
        f3_list.append(f3 if not math.isnan(f3) else replacement_for_nan)
        f4_list.append(f4 if not math.isnan(f4) else replacement_for_nan)

    attributes = dict()

    # Calculate mean formants across pulses
    attributes['f1_mean'] = statistics.mean(f1_list)
    attributes['f2_mean'] = statistics.mean(f2_list)
    attributes['f3_mean'] = statistics.mean(f3_list)
    attributes['f4_mean'] = statistics.mean(f4_list)

    # Calculate median formants across pulses
    attributes['f1_median'] = statistics.median(f1_list)
    attributes['f2_median'] = statistics.median(f2_list)
    attributes['f3_median'] = statistics.median(f3_list)
    attributes['f4_median'] = statistics.median(f4_list)

    # Formant Dispersion (Fitch, W. T. (1997). Vocal tract length and formant frequency
    # dispersion correlate with body size in rhesus macaques. The Journal of the Acoustical
    # Society of America, 102(2), 1213-1222.)
    attributes['formant_dispersion'] = (attributes['f4_median'] -
                                        attributes['f1_median']) / 3

    # Average Formant (Pisanski, K., & Rendall, D. (2011). The prioritization of voice
    # fundamental frequency or formants in listeners’ assessments of speaker size, masculinity,
    # and attractiveness. The Journal of the Acoustical Society of America, 129(4), 2201-2212.)
    attributes['average_formant'] = (attributes['f1_median'] +
                                     attributes['f2_median'] +
                                     attributes['f3_median'] +
                                     attributes['f4_median']) / 4

    # MFF (Smith, D. R., & Patterson, R. D. (2005). The interaction of glottal-pulse rate and
    # vocal-tract length in judgements of speaker size, sex, and age. The Journal of the
    # Acoustical Society of America, 118(5), 3177-3186.)
    attributes['mff'] = (attributes['f1_median'] *
                         attributes['f2_median'] *
                         attributes['f3_median'] *
                         attributes['f4_median']) ** 0.25

    # Fitch VTL (Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion
    # correlate with body size in rhesus macaques. The Journal of the Acoustical Society of
    # America, 102(2), 1213-1222.)
    attributes['fitch_vtl'] = ((1 * (35000 / (4 * attributes['f1_median']))) +
                               (3 * (35000 / (4 * attributes['f2_median']))) +
                               (5 * (35000 / (4 * attributes['f3_median']))) +
                               (7 * (35000 / (4 * attributes['f4_median'])))) / 4

    # Delta F (Reby, D., & McComb, K.(2003). Anatomical constraints generate honesty: acoustic
    # cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.)
    xy_sum = ((0.5 * attributes['f1_median']) +
              (1.5 * attributes['f2_median']) +
              (2.5 * attributes['f3_median']) +
              (3.5 * attributes['f4_median']))
    x_squared_sum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
    attributes['delta_f'] = xy_sum / x_squared_sum

    # VTL(Delta F) Reby, D., & McComb, K.(2003).Anatomical constraints generate honesty: acoustic
    # cues to age and weight in the roars of red deer stags. Animal Behaviour, 65, 519e-530.)
    attributes['vtl_delta_f'] = 35000 / (2 * attributes['delta_f'])

    return attributes#, None

def get_pitch_attributes(filename, pitch_type='preferred', time_step=0., min_time=0., max_time=0.,
                         pitch_floor=75., pitch_ceiling=600., unit='Hertz',
                         interpolation_method='Parabolic', return_values=False,
                         replacement_for_nan=0.):
    """
    Function to get pitch attributes such as minimum pitch, maximum pitch, mean pitch, and
    standard deviation of pitch.

    :param (parselmouth.Sound) sound: sound waveform
    :param (str) pitch_type: the type of pitch analysis to be performed; values include 'preferred'
           optimized for speech based on auto-correlation method, and 'cc' for performing acoustic
           periodicity detection based on cross-correlation method
           NOTE: Praat also includes an option for type 'ac', a variation of 'preferred' that
           requires several more parameters. We are not including this for simplification.
    :param (float) time_step: the measurement interval (frame duration), in seconds (default: 0.)
           NOTE: The default 0. value corresponds to a time step of 0.75 / pitch floor
    :param (float) min_time: minimum time value considered for time range (t1, t2) (default: 0.)
    :param (float) max_time: maximum time value considered for time range (t1, t2) (default: 0.)
           NOTE: If max_time <= min_time, the entire time domain is considered
    :param (float) pitch_floor: minimum pitch (default: 75.)
    :param (float) pitch_ceiling: maximum pitch (default: 600.)
    :param (str) unit: units of the result, 'Hertz' or 'Bark' (default: 'Hertz)
    :param (str) interpolation_method: method of sampling new data points with a discrete set of
           known data points, 'None' or 'Parabolic' (default: 'Parabolic')
    :param (bool) return_values: whether to return a continuous list of pitch values from all frames
           or not
    :param (float) replacement_for_nan: a float number that will represent frames with NaN values
    :return: (a dictionary of mentioned attributes, a list of pitch values OR None)
    """
    sound = parselmouth.Sound(filename)
    # Get total duration of the sound
    duration = call(sound, 'Get end time')

    # Create pitch object
    if pitch_type == 'preferred':
        pitch = call(sound, 'To Pitch', time_step, pitch_floor, pitch_ceiling)
    elif pitch_type == 'cc':
        pitch = call(sound, 'To Pitch (cc)', time_step, pitch_floor, pitch_ceiling)
    else:
        raise ValueError('Argument for @pitch_type not recognized!')

    attributes = dict()
    # Count the number of voiced frames in the window and divide by the total number of frames
    attributes['voiced_fraction'] = call(pitch, 'Count voiced frames') / len(pitch)

    attributes['min_pitch'] = call(pitch, 'Get minimum',
                                   min_time, max_time,
                                   unit,
                                   interpolation_method)

    attributes['relative_min_pitch_time'] = call(pitch, 'Get time of minimum',
                                                 min_time, max_time,
                                                 unit,
                                                 interpolation_method) / duration

    attributes['max_pitch'] = call(pitch, 'Get maximum',
                                   min_time, max_time,
                                   unit,
                                   interpolation_method)

    attributes['relative_max_pitch_time'] = call(pitch, 'Get time of maximum',
                                                 min_time, max_time,
                                                 unit,
                                                 interpolation_method) / duration

    attributes['mean_pitch'] = call(pitch, 'Get mean',
                                    min_time, max_time,
                                    unit)

    attributes['stddev_pitch'] = call(pitch, 'Get standard deviation',
                                      min_time, max_time,
                                      unit)

    attributes['q1_pitch'] = call(pitch, 'Get quantile',
                                  min_time, max_time,
                                  0.25,
                                  unit)

    attributes['median_intensity'] = call(pitch, 'Get quantile',
                                          min_time, max_time,
                                          0.50,
                                          unit)

    attributes['q3_pitch'] = call(pitch, 'Get quantile',
                                  min_time, max_time,
                                  0.75,
                                  unit)

    attributes['mean_absolute_pitch_slope'] = call(pitch, 'Get mean absolute slope', unit)
    attributes['pitch_slope_without_octave_jumps'] = call(pitch, 'Get slope without octave jumps')

    pitch_values = None

    if return_values:
        pitch_values = [call(pitch, 'Get value in frame', frame_no, unit)
                        for frame_no in range(len(pitch))]
        # Convert NaN values to floats (default: 0)
        pitch_values = [value if not math.isnan(value) else replacement_for_nan
                        for value in pitch_values]

    return attributes#, pitch_values



def nPVI(durations):
    """
    Calculate normalized pairwise variability index
    :param durations:
    :return:
    """
    #https://assta.org/proceedings/sst/2006/sst2006-62.pdf
    s = []
    for idx in range(1,len(durations)):
        s.append(float(durations[idx-1]-durations[idx])/float((durations[idx-1]+durations[idx])/2))

    return 100 / float(len(durations)-1) * np.sum(np.abs(s))

def rPVI(durations):
    """
    Calculate raw pairwise variability index
    :param durations:
    :return:
    """

    s = []
    for idx in range(1,len(durations)):
        s.append(float(durations[idx-1]-durations[idx]))

    return np.sum(np.abs(s)) / (len(durations)-1)
