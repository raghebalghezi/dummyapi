import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
from scipy import stats
from scripts.fluency import rPVI, nPVI
import numpy as np
from dataclasses import dataclass


def model_loader(LANG):
        if LANG == 'fi':
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)
        if LANG == 'sv':
                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        return model, processor


model, processor = model_loader("fi")

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

@dataclass
class Point:
    token_index: str
    time_index: int
    score: float

# features from asr: transcript, above, mean_vect_embed

def generate_asr_features(audio_path):

        audio, sample_rate = sf.read(audio_path)

        assert sample_rate == 16_000
        device = torch.device('cpu')
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values

        with torch.no_grad():
                logits = model(input_values.to(device)).logits
                hidden_states = model(input_values.to(device)).hidden_states

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        decoded_ids = processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())

        # number of samples in audio / number of frames (as per w2v2)
        ratio = (len(audio) / logits.shape[1]) #/ sampling_rate

        # list of all (repetative) characters with time_stamps, probs including PAD and |
        path = []

        for idx, j in enumerate(predicted_ids.squeeze()):
        # if decoded_ids[int(idx)] != '<pad>':
                path.append(Point(decoded_ids[int(idx)], (idx+1)*ratio,  torch.softmax(logits.squeeze()[j,:], dim=-1)[j]))

        #pure chars,time stamps and probs
        lst_char = []

        for i,j in enumerate(path):
                if j.token_index != "<pad>":
                        if i<len(path)-1:
                                if j.token_index != path[i+1].token_index:
                                        lst_char.append(j)
                                else:
                                        lst_char.append(j)


        letters = set("abcdefghijklmnopqrstuvwxyz".upper())
        vowels = set("aeiouywåäö".upper()) #a e i o u y å ä ö
        consonants = letters.difference(vowels)
        print(consonants)


        vowels_dur = []
        cons_dur = []
        vowels_prob = []
        cons_prob = []

        for i in lst_char:
                if i.token_index in vowels:
                        vowels_dur.append(i.time_index / 16_000)
                        vowels_prob.append(float(i.score))
                elif i.token_index in consonants:
                        cons_dur.append(i.time_index / 16_000)
                        cons_prob.append(float(i.score))

        acoustic_model_score = -np.log((np.sum(vowels_prob) + np.sum(cons_prob))) / len(transcription[0].split(" "))

        vwl_dur_stat = stats.describe(vowels_dur)
        cons_dur_stat = stats.describe(cons_dur)
        vwl_prob_stat = stats.describe(-np.log(vowels_prob))
        cons_prob_stat = stats.describe(-np.log(cons_prob))

        feature_dict = dict()
        feature_dict['transcript'] = transcription[0]
        feature_dict['acoustic_model_score'] = acoustic_model_score
        feature_dict['cons_dur_kur'] = cons_dur_stat.kurtosis
        feature_dict['cons_dur_max']  = cons_dur_stat.minmax[1]
        feature_dict['cons_dur_mean'] = cons_dur_stat.mean
        feature_dict['cons_dur_min'] = cons_dur_stat.minmax[0]
        feature_dict['cons_dur_skew'] = cons_dur_stat.skewness
        feature_dict['cons_dur_var'] = cons_dur_stat.variance
        feature_dict['cons_prob_kur'] = cons_prob_stat.kurtosis
        feature_dict['cons_prob_mean'] = cons_prob_stat.mean
        feature_dict['cons_prob_max'] = cons_prob_stat.minmax[1]
        feature_dict['cons_prob_min'] = cons_prob_stat.minmax[0]
        feature_dict['cons_prob_skew'] = cons_prob_stat.skewness
        feature_dict['cons_prob_var'] = cons_prob_stat.variance
        feature_dict['nPVI_cons'] = nPVI(cons_dur)
        feature_dict['nPVI_vowels'] = nPVI(vowels_dur)
        feature_dict['rPVI_cons'] = rPVI(cons_dur)
        feature_dict['rPVI_vowels'] = rPVI(vowels_dur)
        feature_dict['vowels_dur_kur'] = vwl_dur_stat.kurtosis
        feature_dict['vowels_dur_max'] = vwl_dur_stat.minmax[1]
        feature_dict['vowels_dur_mean'] = vwl_dur_stat.mean
        feature_dict['vowels_dur_min'] = vwl_dur_stat.minmax[0]
        feature_dict['vowels_dur_skew'] = vwl_dur_stat.skewness
        feature_dict['vowels_dur_var'] = vwl_dur_stat.variance
        feature_dict['vowels_prob_kur'] = vwl_prob_stat.kurtosis
        feature_dict['vowels_prob_max'] = vwl_prob_stat.minmax[1]
        feature_dict['vowels_prob_mean'] = vwl_prob_stat.mean
        feature_dict['vowels_prob_min'] = vwl_prob_stat.minmax[0]
        feature_dict['vowels_prob_skew'] = vwl_prob_stat.skewness
        feature_dict['vowels_prob_var'] =  vwl_prob_stat.variance

        feature_dict['embedding'] = hidden_states[-1].squeeze().mean(axis=0).detach().numpy()

        return feature_dict



print(generate_asr_features("8_1_joens_3hv_16.wav"))


# print("AM score", acoustic_model_score)
# # summary stat of vowel durations in each sample
# print(stats.describe(vowels_dur))
# # summary stat of cons durations in each sample
# print(stats.describe(cons_dur))

# summary stat of vowel -ll in each sample
# print(stats.describe(-np.log(vowels_prob)))
# # summary stat of cons -ll in each sample
# print(stats.describe(-np.log(cons_prob)))

# print("Row Pairwise Vocalic Indexing", rPVI(vowels_dur))
# print("Normalized Pairwise Vocalic Indexing", nPVI(vowels_dur))

# print("Row Pairwise Vocalic Indexing", rPVI(cons_dur))
# print("Normalized Pairwise Vocalic Indexing", nPVI(cons_dur))
