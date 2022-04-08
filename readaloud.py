import soundfile as sf
import torch
from jiwer import cer, wer
from utils import model_loader, make_tag



model, processor = model_loader('fi')
  


def generate_readaloud(audio_path, prompt, model, processor):

  audio, sample_rate = sf.read(audio_path)
  prompt = prompt.upper()

  device = torch.device('cpu')
  input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
  with torch.no_grad():
      logits = model(input_values.to(device)).logits
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.batch_decode(predicted_ids)

  gop_score = 1 - cer(prompt, transcription[0])

  results = dict()
  results['gop'] = gop_score

  a = prompt
  b = transcription[0]

  res = make_tag(a, b)
  results['annnotated_response'] = res

  return results

