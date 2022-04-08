from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from difflib import SequenceMatcher

def model_loader(LANG):
  if LANG == 'fi':
      model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
      processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
  if LANG == 'sv':
      model =  Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
      processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
  return model, processor


def make_tag(a: str, b: str) -> str:
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