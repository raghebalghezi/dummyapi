from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from scipy import stats
import numpy as np
import torch
from dataclasses import dataclass

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

@dataclass
class Point:
    token_index: str
    time_index: int
    score: float

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

input_value = dataset[0]["audio"]["array"]

# audio file is decoded on the fly
inputs = processor(input_value, sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# number of samples in audio / number of frames (as per w2v2)
ratio = (len(input_value) / logits.shape[1]) #/ sampling_rate

decoded_ids = processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())

# transcribe speech
transcription = processor.batch_decode(predicted_ids)[0]
print(transcription)

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
vowels = set("aeiouyw".upper())
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

acoustic_model_score = -np.log((np.sum(vowels_prob) + np.sum(cons_prob))) / len(transcription.split(" "))

print("AM score", acoustic_model_score)
# summary stat of vowel durations in each sample
print(stats.describe(vowels_dur))
# summary stat of cons durations in each sample
print(stats.describe(cons_dur))

# summary stat of vowel -ll in each sample
print(stats.describe(-np.log(vowels_prob)))
# summary stat of cons -ll in each sample
print(stats.describe(-np.log(cons_prob)))

print("Row Pairwise Vocalic Indexing", rPVI(vowels_dur))
print("Normalized Pairwise Vocalic Indexing", nPVI(vowels_dur))

print("Row Pairwise Vocalic Indexing", rPVI(cons_dur))
print("Normalized Pairwise Vocalic Indexing", nPVI(cons_dur))
