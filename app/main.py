# import os
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from gensim import corpora, models
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import nltk


# nltk.download('stopwords')

# current_dir = os.path.dirname(os.path.abspath(__file__))

# def load_ldamodel():

#     ldamodel_path = os.path.join(current_dir, "lda_model100")
#     ldamodel = models.ldamodel.LdaModel.load(ldamodel_path)

#     return ldamodel

# # def load_ldamodel():
# #     ldamodel = models.ldamodel.LdaModel.load(os.path.join(current_dir, "..", "lda_model100"))
# #     return ldamodel

# ldamodel = load_ldamodel()

# def preprocess(text):
#     tokenizer = RegexpTokenizer(r'\w+')
#     en_stop = set(stopwords.words('english'))
#     p_stemmer = PorterStemmer()
    
#     raw = text.lower()
#     tokens = tokenizer.tokenize(raw)
#     stopped_tokens = [i for i in tokens if not i in en_stop]
#     stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
#     return stemmed_tokens

# def infer_topics(new_sentence):
#     preprocessed_sentence = preprocess(new_sentence)
#     dictionary = ldamodel.id2word
#     bow = dictionary.doc2bow(preprocessed_sentence)
#     topic_distribution = ldamodel.get_document_topics(bow)

#     if topic_distribution:
#         most_probable_topic = max(topic_distribution, key=lambda x: x[1])
#         most_probable_topic_id, prob = most_probable_topic
#         prob = float(prob)
#         return most_probable_topic_id, prob
#     else:
#         return -1, 0.0


# def find_matching_sura_ayah(most_probable_topic, filename="your_file_with_topics_lda100.csv"):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     csv_path = os.path.join(current_dir, "files", filename)
#     df = pd.read_csv(csv_path)
#     df['sura_ayah'] = df['sura'].astype(str) + ':' + df['ayah'].astype(str)
#     matching_sura_ayah = df[df['topic'] == most_probable_topic]['sura_ayah'].tolist()
#     return matching_sura_ayah

# class Sentence(BaseModel):
#     text: str

# app = FastAPI()

# @app.post("/infer/")
# async def infer_topics_api(sentence: Sentence):
#     most_probable_topic, prob = infer_topics(sentence.text)
#     if most_probable_topic != -1:
#         matching_sura_ayah = find_matching_sura_ayah(most_probable_topic)
#         return {"sura_ayah": matching_sura_ayah, "probability": prob}
#     else:
#         raise HTTPException(status_code=404, detail="No dominant topic with probability over 50% was found for this sentence.")





import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import wordnet # import wordnet module

nltk.download('stopwords')
nltk.download('wordnet') # download wordnet data


current_dir = os.path.dirname(os.path.abspath(__file__))

# nltk.data.path.append(os.path.join(current_dir, 'wordnet.zip'))
# nltk.data.path.append(os.path.join(current_dir, 'stopwords.zip'))

def load_ldamodel():

    ldamodel_path = os.path.join(current_dir, "lda_model100")
    ldamodel = models.ldamodel.LdaModel.load(ldamodel_path)

    return ldamodel

ldamodel = load_ldamodel()

def preprocess(text):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    return stemmed_tokens

def infer_topics(new_sentence):
    preprocessed_sentence = preprocess(new_sentence)
    dictionary = ldamodel.id2word
    bow = dictionary.doc2bow(preprocessed_sentence)
    topic_distribution = ldamodel.get_document_topics(bow)

    if topic_distribution:
        most_probable_topic = max(topic_distribution, key=lambda x: x[1])
        most_probable_topic_id, prob = most_probable_topic
        prob = float(prob)
        return most_probable_topic_id, prob
    else:
        return -1, 0.0
        

def find_matching_sura_ayah(most_probable_topic, filename="your_file_with_topics_lda100.csv"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "files", filename)
    df = pd.read_csv(csv_path)
    df['sura_ayah'] = df['sura'].astype(str) + ':' + df['ayah'].astype(str)
    matching_sura_ayah = df[df['topic'] == most_probable_topic]['sura_ayah'].tolist()
    return matching_sura_ayah

class Sentence(BaseModel):
    text: str

app = FastAPI()

# Define a function to get synonyms of a word
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return list(set(synonyms)) # remove duplicates

@app.post("/infer/")
async def infer_topics_api(sentence: Sentence):
    # Get the top 5 synonyms of the input word
    synonyms = get_synonyms(sentence.text)[:5]
    # Initialize an empty dictionary to store the results
    results = {"words": [], "refs": []}
    # Loop through the synonyms
    for synonym in synonyms:
        # Apply the infer_topics function and the find_matching_sura_ayah function
        most_probable_topic, prob = infer_topics(synonym)
        if most_probable_topic != -1:
            # Append the synonym to the "words" list
            results["words"].append(synonym)
            # Append the matching sura:ayah data to the "refs" list
            matching_sura_ayah = find_matching_sura_ayah(most_probable_topic)
            results["refs"].extend(matching_sura_ayah)
    # Return the results as a JSON object
    return results
