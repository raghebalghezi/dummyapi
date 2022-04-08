from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def model_loader(LANG):
  if LANG == 'fi':
      model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
      processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
  if LANG == 'sv':
      model =  Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
      processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
  return model, processor


