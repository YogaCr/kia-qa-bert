import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import streamlit as st
from annotated_text import annotated_text

from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import glob

import numpy as np
import pandas as pd

import string
import re
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from datetime import datetime

hour = datetime.now().hour
greeting = ""
if hour < 12:
    greeting = "Selamat Pagi"
elif hour < 15:
    greeting = "Selamat Siang"
elif hour < 18:
    greeting = "Selamat Sore"
else:
    greeting = "Selamat Malam"

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)

st.set_page_config(page_title="Tanya Jawab Kesehatan Ibu dan Anak")

@st.cache_resource
def import_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

@st.cache_resource
def import_tokenizer():
    return BertTokenizer.from_pretrained("YogaCr/kia-qa-model")

@st.cache_resource(show_spinner=False)
def import_model():
    return BertForQuestionAnswering.from_pretrained("YogaCr/kia-qa-model")

tokenizer = import_tokenizer()
model = import_model()
model = model.to(torch_device)
model.eval()

kb_datas = st.session_state.get('kb_datas', pd.DataFrame(columns=['context','tokenized','file_path']))

@st.cache_data
def import_kb_files():
    return glob.glob("./md-informasi-buku-kia/reformatted-text/*/*.md")

kb_files = import_kb_files()
    
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

@st.cache_data
def import_context():
    kb_datas = pd.DataFrame(columns=['context','tokenized','file_path'])
    for path in kb_files:
        f = open(path, "r")
        context = f.read()
        
        stemmed_tokens = preprocess_text(context)
        split_path = path.split('/')
        formatted_path = split_path[-2]+"/"+split_path[-1]
        kb_datas = pd.concat([kb_datas, pd.DataFrame({'context':[context], 'tokenized':[' '.join(stemmed_tokens)],'file_path':formatted_path})], ignore_index=True)
    st.session_state.kb_datas = kb_datas

def choose_context(question):
    kb_datas = st.session_state.get('kb_datas', pd.DataFrame(columns=['context','tokenized','file_path']))
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
    print(similar_doc)
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
                if (word in string.punctuation and (word != '-' and word != '/')):
                    print(word)
                final_answer += (' ' if i>0 and (not final_answer[-1] in string.punctuation or (final_answer[-1] in string.punctuation and (final_answer[-1] != '-' and final_answer[-1] != '/'))) else '')+ word
            i += 1
    print(final_answer)
    torch.cuda.empty_cache()
    gc.collect()
    return final_answer,context, file_path, total_score, start_index, end_index

def getanswer(question: str,show_rank: bool = False):
    summ,context, path, score,start_index, end_index = qa_system(question)
    if summ == "":
        summ = "Maaf, kami tidak dapat menemukan jawaban dari pertanyaan anda"
        path = ""
    return {"answer": summ,"context":context, "path": path, "score": score, "start_index": start_index, "end_index": end_index}

def getmd(file_path: str):
    try:
        f = open("./md-informasi-buku-kia/"+file_path, "r")
        context = f.read()
        return {"content": context}
    except:
        return {"content": "Not Found"}
    
st.title("KIA Question Answering System")
data_load_state = st.text('Loading data...')
import_nltk()
import_context()
print("startup done")

data_load_state.text('')
question = st.text_input('%s, selamat datang di aplikasi sistem tanya jawab untuk kesehatan ibu dan anak. Apabila anda memiliki pertanyaan seputar kesehatan ibu dan anak, anda dapat mengisinya pada kotak di bawah.'%(greeting), placeholder = 'Berapa lama jam tidur malam ibu hamil?')
if st.button("Submit"):
    
    answer= getanswer(question)
    
    if answer['score'] > 0 :
        answer_text=re.sub(r'\s+(?=[\W])|(?<=[-])\s', '', answer['answer'])
        answer_context = re.sub(r'\s+(?=[\W])|(?<=[-])\s', '', answer['context'])   

        next_sentence_index = get_next_sentence(question, answer_context)
        next_sentence = [re.split(r'[.;?!]',answer_context)[x] for x in next_sentence_index] if len(next_sentence_index)>0 else []
        is_in_next_sentence = True
        
        for v in next_sentence:
            if answer_text.lower() in v.lower():
                is_in_next_sentence = False
                break
        if is_in_next_sentence:
            next_sentence.insert(0,answer_text)
        st.write("Jawaban: ", ". ".join(next_sentence) if len(next_sentence)>0 else answer_text)
    else: 
        st.write("Jawaban: ", answer['answer'])
    
    if answer['score'] > 0 :
        answer_index, end_index = -1,-1

        for i in re.finditer(answer_text.lower(), answer_context.lower()):
            answer_index = i.start()
            end_index = i.end()
            break
        
        st.write("Sumber Jawaban: ")
        annotated_text(
            answer_context[:answer_index],
            (answer_context[answer_index:end_index], "","#f39c12"),
            answer_context[end_index:]
        )

        st.write("Teks Asli (Tanpa parafrase): ")
        st.markdown(getmd(answer['path'])['content'])

        st.write("Skor: ", answer['score'])
