import pandas as pd
import numpy as np

pronun_features = ['acoustic_model_score','cons_dur_kur', 'cons_dur_max', 
        'cons_dur_mean', 'cons_dur_min',
       'cons_dur_skew', 'cons_dur_var', 'cons_prob_kur', 'cons_prob_max',
       'cons_prob_mean', 'cons_prob_min', 'cons_prob_skew', 'cons_prob_var', 'nPVI_cons', 
       'nPVI_vowels', 'rPVI_cons',
       'rPVI_vowels', 'sample', 'vowels_dur_kur',
       'vowels_dur_max', 'vowels_dur_mean', 'vowels_dur_min',
       'vowels_dur_skew', 'vowels_dur_var', 'vowels_prob_kur',
       'vowels_prob_max', 'vowels_prob_mean', 'vowels_prob_min',
       'vowels_prob_skew', 'vowels_prob_var']

fluen_features = ['sample', 'nsyll', 'npause',
       'dur(s)', 'phonationtime(s)', 'speechrate(nsyll / dur)',
       'articulation rate(nsyll / phonationtime)', 'ASD(speakingtime / nsyll)',
       'voiced_fraction', 'min_pitch', 'relative_min_pitch_time', 'max_pitch',
       'relative_max_pitch_time', 'mean_pitch', 'stddev_pitch', 'q1_pitch',
       'median_intensity', 'q3_pitch', 'mean_absolute_pitch_slope',
       'pitch_slope_without_octave_jumps', 'f1_mean', 'f2_mean', 'f3_mean',
       'f4_mean', 'f1_median', 'f2_median', 'f3_median', 'f4_median',
       'formant_dispersion', 'average_formant', 'mff', 'fitch_vtl', 'delta_f',
       'vtl_delta_f', 'sequence_length', 'perplexity_nonnative', 'perplexity_native']




RANDOM_SEED = 89

def true_round(x):
    import decimal
    return int(decimal.Decimal(str(x)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))


pron_data = pd.read_csv('/Volumes/scratch/work/algher1/LAQ_Data/swedish_pronunciation_features.csv', usecols=pronun_features)

swedishfluen_data = pd.read_csv("/Volumes/scratch/work/algher1/LAQ_Data/swedish_acoustic_features.csv", usecols=fluen_features)

gram_data = pd.read_csv('/Volumes/scratch/work/algher1/LAQ_Data/swedish_asr_text_features.csv')

lexical_data = pd.read_csv('../swedish/swedish_lexical_asr.csv', usecols=['sample', 'length_s',
       'length_w', 'types', 'TTR', 'rootTTR', 'correctedTTR', 'logTTR', 'ovix',
       'tfidf', 'mostfrequentwords', 'leastfrequentwords'])

holistic_labels = pd.read_csv("/Volumes/scratch/work/algher1/LAQ_Data/swedish_fairAvg_combined.csv", usecols=['Sample','holistic_FairAvg'])
holistic_labels.rename(columns={'Sample': 'sample'}, inplace=True)
holistic_labels['sample'] = holistic_labels['sample'].astype('int64')

from functools import reduce
df = reduce(lambda df1,df2: pd.merge(df1,df2,on='sample'), [pron_data, swedishfluen_data, gram_data, lexical_data, holistic_labels])

df.drop(columns=['sample', "Unnamed: 0", "identifier","n_sentences"], inplace=True)
df.fillna(0.0, inplace=True)

# drop sparse features
df = df.loc[:, (df != 0).any(axis=0)]
# pron_labels['pronunciation_FairAvg'] = pron_labels['pronunciation_FairAvg'].map(np.vectorize(true_round))
