from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import glob

import numpy as np
import pandas as pd

import string
import re
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import gc

app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)

tokenizer = BertTokenizer.from_pretrained("YogaCr/kia-qa-model")
model = BertForQuestionAnswering.from_pretrained("YogaCr/kia-qa-model")
model = model.to(torch_device)
model.eval()

kb_datas = pd.DataFrame(columns=['context','tokenized'])

kb_files = glob.glob("../md-informasi-buku-kia/reformatted-text/*/*.md")
    
def preprocess_text(context):
    lowercase_context = context.lower()
    lowercase_context = lowercase_context.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
    # print(lowercase_context)
    lowercase_context = lowercase_context.strip()
    lowercase_context = re.sub(r'\s+', ' ', lowercase_context)

    tokens = word_tokenize(lowercase_context)
    # print(tokens)
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    # print(filtered_tokens)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    # print(stemmed_tokens)
    return stemmed_tokens

for path in kb_files:
    f = open(path, "r")
    context = f.read()
    
    stemmed_tokens = preprocess_text(context)
    split_path = path.split('/')
    formatted_path = split_path[-2]+"/"+split_path[-1]
    kb_datas = pd.concat([kb_datas, pd.DataFrame({'context':[context], 'tokenized':[' '.join(stemmed_tokens)],'file_path':formatted_path})], ignore_index=True)

print("ready")

def choose_context(question):
    query_tokens = preprocess_text(question)
    query_tokens = ' '.join(query_tokens)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(kb_datas['tokenized'])
    query_vector = vectorizer.transform([query_tokens])
    scores = np.dot(X, query_vector.T).toarray().flatten()
    best_context_index = np.where(scores==np.max(scores))[0][0]
    return kb_datas['context'][best_context_index], kb_datas['file_path'][best_context_index]

def get_next_sentence(question, context):
    query_tokens = preprocess_text(question)
    context_splitted = re.split(r'[.;?!]', context)
    context_tokens = [preprocess_text(x) for x in context_splitted if x != '']

    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(context_tokens)]
    model = Doc2Vec(tagged_data, min_count = 2, workers = 4, epochs=100)

    test_doc = word_tokenize(' '.join(query_tokens))
    test_doc_vector = model.infer_vector(test_doc)
    similar_doc = model.dv.most_similar([test_doc_vector])
    selected = [int(x[0]) for x in similar_doc if x[1]>=0.5]
    selected.sort()
    return selected

def qa_system(question, max_len=512):
    context, file_path = choose_context(question)
    
    inputs = tokenizer.encode_plus(question, context)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    # Get the start and end index of the answer
    model_res = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    start_logits = model_res.start_logits
    end_logits = model_res.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1
    total_score = (start_logits[0, start_index].item() + end_logits[0, end_index].item())
    
    # Get the answer
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    final_answer = ""
    if(total_score>0):
        answer = ' '.join(tokens[start_index:end_index])
        i = 0
        for word in answer.split():
            if word[0:2] == '##':
                final_answer += word[2:]
            else:
                final_answer += (' ' if i>0 else '')+ word
            i += 1
    torch.cuda.empty_cache()
    gc.collect()
    return final_answer,context, file_path, total_score
class QARequest(BaseModel):
    question: str

@app.get('/')
async def home():
    return {"message": "Hello World"}

@app.post("/qa")
async def getanswer(qa_req_body: QARequest,show_rank: bool = False):
    summ,context, path, score = qa_system(qa_req_body.question)

    answer_text=re.sub(r'\s+(?=[\W])|(?<=[-])\s', '', summ)
    answer_context = re.sub(r'\s+(?=[\W])|(?<=[-])\s', '', context)   

    next_sentence_index = get_next_sentence(qa_req_body.question, answer_context)
    print(next_sentence_index)
    next_sentence = [re.split(r'[.;?!]',answer_context)[x] for x in next_sentence_index] if len(next_sentence_index)>0 else []
    is_in_next_sentence = True
    for v in next_sentence:
        if answer_text.lower() in v.lower():
            is_in_next_sentence = False
            break
    if is_in_next_sentence:
        next_sentence.insert(0,answer_text)
    if summ == "":
        summ = "Maaf, kami tidak dapat menemukan jawaban dari pertanyaan anda"
        path = ""
    return {"answer": (". ".join(next_sentence) if len(next_sentence)>0 else answer_text).replace("  "," ").strip(),"context":context, "path": path, "score": score}
    # return {"answer":  summ,"context":context, "path": path, "score": score}

@app.get("/md")
async def getmd(file_path: str):
    try:
        f = open("../md-informasi-buku-kia/"+file_path, "r")
        context = f.read()
        return {"content": context}
    except:
        return {"content": "Not Found"}

