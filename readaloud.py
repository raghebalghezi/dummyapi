import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import cer, wer
from difflib import SequenceMatcher

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

audio, sample_rate = sf.read("/content/must_apologize.wav")
prompt = "i musta poll gize for dragging you all heir in such an un common hour".upper()
device = torch.device('cpu')
input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
with torch.no_grad():
    logits = model(input_values.to(device)).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
# return transcription[0]

gop_score = 1 - cer(prompt, transcription[0])

a = prompt.split()
b = transcription[0].split()


def make_tag(a, b):
  new = "<p style='color: green'>"
  s = SequenceMatcher(None, a, b)
  for tag, i1, i2, j1, j2 in s.get_opcodes():
    if tag == 'equal':
      new += a[i1:i2]
    elif tag ==  "delete":
      new += f"<del style='color: red'>{a[i1:i2]}</del>"
    elif tag == "insert":
      new += f"<ins style='color: orange'>{b[j1:j2]}</ins>"
    elif tag == "replace":
      new += f"<del style='color: red'>{a[i1:i2]}</del>"
      new += f"<ins style='color: orange'>{b[j1:j2]}</ins>"
  new += "</p>"
  return new

reponse = make_tag(a,  b)

print("gop", gop_score)

# replace   a[1:4] --> b[1:3] ['MUSTA', 'POLL', 'GIZE'] --> ['MUST', 'APOLOGIZE']
# replace   a[8:9] --> b[7:8] ['HEIR'] --> ['HERE']
# replace   a[12:13] --> b[11:12]   ['UN'] --> ['A']
# gop 0.8840579710144928

# insert    a[6:6] --> b[6:7]       '' --> ' '
# delete    a[7:8] --> b[8:8]      ' ' --> ''
# replace   a[11:13] --> b[11:12]     'L ' --> 'O'
# delete    a[41:42] --> b[40:40]      'I' --> ''
# insert    a[43:43] --> b[41:42]       '' --> 'E'
# replace   a[55:57] --> b[54:55]     'UN' --> 'A'
# gop 0.8840579710144928