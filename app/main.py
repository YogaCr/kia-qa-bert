import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import BertTokenizer, BertForQuestionAnswering
import torch

from rank_bm25 import BM25Okapi

import glob

import numpy as np
import pandas as pd

import string
import re
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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
    lowercase_context = lowercase_context.translate(str.maketrans('', '', string.punctuation))
    lowercase_context = lowercase_context.strip()
    lowercase_context = re.sub(r'\s+', ' ', lowercase_context)

    tokens = word_tokenize(lowercase_context)

    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [w for w in tokens if not w in stop_words]

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    return stemmed_tokens

for path in kb_files:
    f = open(path, "r")
    context = f.read()
    
    stemmed_tokens = preprocess_text(context)
    split_path = path.split('/')
    formatted_path = split_path[-2]+"/"+split_path[-1]
    kb_datas = pd.concat([kb_datas, pd.DataFrame({'context':[context], 'tokenized':[' '.join(stemmed_tokens)],'file_path':formatted_path})], ignore_index=True)

print("ready")

class QARequest(BaseModel):
    question: str


def choose_context(question):
    query_tokens = preprocess_text(question)
    query_tokens = ' '.join(query_tokens)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(kb_datas['tokenized'])
    query_vector = vectorizer.transform([query_tokens])
    scores = np.dot(X, query_vector.T).toarray().flatten()
    best_context_index = np.where(scores==np.max(scores))[0][0]
    return kb_datas['context'][best_context_index], kb_datas['file_path'][best_context_index]

def qa_system(question, max_len=512):
    context, file_path = choose_context(question)
    
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    # Get the start and end index of the answer
    model_res = model(input_ids, token_type_ids=token_type_ids)
    start_logits = model_res.start_logits
    end_logits = model_res.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1
    total_score = (start_logits[0, start_index].item() + end_logits[0, end_index].item())*10
    
    # Get the answer
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer=""
    if(total_score>0):
        answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index])
    torch.cuda.empty_cache()
    gc.collect()
    return answer,context, file_path, total_score


@app.get('/')
async def home():
    return {"message": "Hello World"}

@app.post("/qa")
async def getanswer(qa_req_body: QARequest,show_rank: bool = False):
    summ,context, path, score = qa_system(qa_req_body.question)
    if summ == "":
        summ = "Maaf, kami tidak dapat menemukan jawaban dari pertanyaan anda"
        path = ""
    return {"answer": summ,"context":context, "path": path, "score": score}

@app.get("/md")
async def getmd(file_path: str):
    print(file_path)
    try:
        f = open("../md-informasi-buku-kia/"+file_path, "r")
        context = f.read()
        return {"content": context}
    except:
        return {"content": "Not Found"}