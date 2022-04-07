from scripts.fluency import (speech_rate, get_formant_attributes, 
                                                get_pitch_attributes)


feature_selected = ['nsyll', 'npause',
       'dur(s)', 'phonationtime(s)', 'speechrate(nsyll / dur)',
       'articulation rate(nsyll / phonationtime)', 'ASD(speakingtime / nsyll)',
       'voiced_fraction', 'min_pitch', 'relative_min_pitch_time', 'max_pitch',
       'relative_max_pitch_time', 'mean_pitch', 'stddev_pitch', 'q1_pitch',
       'median_intensity', 'q3_pitch', 'mean_absolute_pitch_slope',
       'pitch_slope_without_octave_jumps', 'f1_mean', 'f2_mean', 'f3_mean',
       'f4_mean', 'f1_median', 'f2_median', 'f3_median', 'f4_median',
       'formant_dispersion', 'average_formant', 'mff', 'fitch_vtl', 'delta_f',
       'vtl_delta_f']

def generate_fluency_features(audio):
    total_fluency_features = dict()
    speech_rate_info = speech_rate(audio)
    formants_info = get_formant_attributes(audio)
    pitch_info = get_pitch_attributes(audio)
    total_fluency_features.update(speech_rate_info)
    total_fluency_features.update(formants_info)
    total_fluency_features.update(pitch_info)


    return total_fluency_features

# feat_dict = generate_fluency_features('8_1_joens_3hv_16.wav')

# print(set(feat_dict.keys()).difference(set(feature_selected)))
# print(feat_dict)