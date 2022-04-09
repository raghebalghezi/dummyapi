from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')


def sim(transcript, prompt):
    transcript_embedding = model.encode(transcript)
    prompt_embedding = model.encode(prompt)

    return float(util.dot_score(transcript_embedding, prompt_embedding).numpy())