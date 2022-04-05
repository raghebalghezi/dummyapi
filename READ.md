ASR: 
    fin_w2v2
    Sv_w2v2
Features:
    Acoustic/Pronunciation feature scripts
    Lexical features  only
Trained Models:
    Holistic: Decision Tree trained on selective features
    TA: 

```
def asr_wav2vec(audio, sr, model, processor):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    #model = model.to(device)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]


def asr(kaldi_path, w2v_model=None, w2v_processor=None, audio=None, sr=None):
    if audio is None and sr is None:
        print('Error! No audio loaded!')
        #audio, sr = get_audio() #a function for recording audio
    elif type(audio) is str:
        audio, sr = librosa.load(audio, sr=16000)
    if sr != 16000:
        #audio = librosa.resample(np.array(audio/32767.0, dtype=np.float32), sr, 16000)
        audio = librosa.resample(audio, sr, 16000)
        sr = 16000

    transcript_kaldi=asr_kaldi(kaldi_path=kaldi_path, audio=audio, sr=sr)
    #transcript_wav2vec=asr_wav2vec(audio=audio, sr=sr, model=w2v_model, processor=w2v_processor)
    return transcript_kaldi#, transcript_wav2vec
```


Readaloud: F(prompt, audio, lang) -> annotated_prompt, gop

Freeform: (0) Featurizer(transcript) -> Dict(lexical, grammatical)
        (1) Featurizer(audio) -> Dict(fluency, pronunciation)
        (2) W2v2 Embeddings -> mean vector of 1024

Models:
    Train,test and pickle DNNs for holistic, pron, flu, lexicogram
    TA: unsuper

Stress  Test: ~ 30 students