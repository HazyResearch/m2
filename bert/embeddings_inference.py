
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import os

from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_model, save_model 
import pdb
import torch
from tabulate import tabulate

from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from typing import Optional, cast
from datetime import datetime

import torch
import torch.nn.functional as F

import src.create_bert as bert_module
from src.embeddings.create_LoCo import load_multi_news, load_pubmed_qa, load_tau_fs, load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum 
from src.embeddings.create_LoCo import load_qasper, load_trivia_qa, load_kilt_dataset, load_long_bench, load_tau_scrolls_needle

import time
import openai
from sentence_transformers import SentenceTransformer
import requests

import voyageai 
from voyageai import get_embeddings

import cohere

################################################################

def expand_tensor(input_tensor, cfg):
    target_size = (1, cfg.get("evaluation_max_seq_len"))
    padding = [0, target_size[1] - input_tensor.size(1)]
    expanded_tensor = F.pad(input_tensor, padding)
    assert expanded_tensor.shape[1] == target_size[1]
    return expanded_tensor

################################################################

import tiktoken
def cutoff_long_text_for_embedding_generation(text, encoding, cutoff=8192):
    encoded_text = encoding.encode(text)[:cutoff]
    decoded_text = encoding.decode(encoded_text)
    return decoded_text

def split_long_text_for_embedding_generation(text, encoding, cutoff=8192):
    encoded_texts = split_list(encoding.encode(text), cutoff)
    decoded_texts = [encoding.decode(encoded_text) for encoded_text in encoded_texts]
    return decoded_texts[:4]

################################################################

def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

def get_embedding(text, encoding, model="text-embedding-ada-002"):
    for _ in range(5):
        try:
            text = cutoff_long_text_for_embedding_generation(text, encoding, cutoff=8192)
            return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
        except:
            print("Error generating embedding! Attempting again...")
            time.sleep(30)

################################################################
    
#############################################

class OpenAI_Encoder:
    def __init__(self, embedding_model="text-embedding-ada-002", **kwargs):
        self.embedding_model = embedding_model
        self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
        self.encoder_batch_size = 256

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        queries = [cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=8192) for query in queries]
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size)):
            try:
                encoded_queries = openai.Embedding.create(input=query_chunks, model=self.embedding_model)
            except:
                time.sleep(30)
                encoded_queries = openai.Embedding.create(input=query_chunks, model=self.embedding_model)
            encoded_queries = [query_encoding['embedding'] for query_encoding in encoded_queries['data']]
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

        
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        passages = [cutoff_long_text_for_embedding_generation(passage, self.encoding, cutoff=8192) for passage in passages]
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size)):
            try:
                encoded_passages = openai.Embedding.create(input=passage_chunks, model=self.embedding_model)
            except:
                time.sleep(30)
                encoded_passages = openai.Embedding.create(input=passage_chunks, model=self.embedding_model)
            encoded_passages = [passage_encoding['embedding'] for passage_encoding in encoded_passages['data']]
            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)
        
#############################################

class Voyager_Encoder:
    def __init__(self, embedding_model="voyage-01", **kwargs):
        self.embedding_model = embedding_model
        #self.encoding = tiktoken.encoding_for_model("voyage-01")
        self.encoder_batch_size = 64

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        #queries = [cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=4096) for query in queries]
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size)):
            try:
                encoded_queries = get_embeddings(query_chunks, model=self.embedding_model, input_type="query")
            except:
                time.sleep(30)
                encoded_queries = get_embeddings(query_chunks, model=self.embedding_model, input_type="query")
            #encoded_queries = [query_encoding for query_encoding in encoded_queries]
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

        
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        #passages = [cutoff_long_text_for_embedding_generation(passage, self.encoding, cutoff=8192) for passage in passages]
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size)):
            try:
                encoded_passages = get_embeddings(passage_chunks, model=self.embedding_model, input_type="document")
            except:
                time.sleep(30)
                encoded_passages = get_embeddings(passage_chunks, model=self.embedding_model, input_type="document")
            encoded_passages = [passage_encoding for passage_encoding in encoded_passages]
            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)
        
#############################################

class Cohere_Encoder:
    def __init__(self, truncation, embedding_model="embed-english-v3.0",  **kwargs):
        self.embedding_model = embedding_model
        #self.encoding = tiktoken.encoding_for_model(embedding_model)
        self.encoder_batch_size = 64
        self.co = cohere.Client(os.environ["COHERE_API_KEY"])
        self.truncation = truncation

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        #queries = [cutoff_long_text_for_embedding_generation(query, self.encoding, cutoff=4096) for query in queries]
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size)):
            try:
                encoded_queries = self.co.embed(texts=query_chunks, model=self.embedding_model, input_type="search_query", truncate=self.truncation)
            except:
                time.sleep(30)
                encoded_queries = self.co.embed(texts=query_chunks, model=self.embedding_model, input_type="search_query", truncate=self.truncation)
            encoded_queries = encoded_queries.embeddings
            total_encoded_queries += encoded_queries
        return np.array(total_encoded_queries)

        
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        #passages = [cutoff_long_text_for_embedding_generation(passage, self.encoding, cutoff=8192) for passage in passages]
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size)):
            try:
                encoded_passages = self.co.embed(texts=passage_chunks, model=self.embedding_model, input_type="search_document", truncate=self.truncation)
            except:
                time.sleep(30)
                encoded_passages = self.co.embed(texts=passage_chunks, model=self.embedding_model, input_type="search_document", truncate=self.truncation)
            encoded_passages = encoded_passages.embeddings
            total_encoded_passages += encoded_passages
        return np.array(total_encoded_passages)
    
#############################################

class M2_BERT_Encoder:
    def __init__(self, checkpoint=None, cfg=None, **kwargs):

        self.cfg = cfg

        ######################################

        model = bert_module.create_bert_classification(
            num_labels=10, # Label value is insignificant since classifier is dropped
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=None,
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            gradient_checkpointing=cfg.get('gradient_checkpointing', None)
        ).eval()

        #################################

        if "pytorch" in cfg.get("pretrained_checkpoint") or ".pt" in cfg.get("pretrained_checkpoint"):
            state_dict = torch.load(cfg.get('pretrained_checkpoint'))#['module']
            new_dict = {}
            for key in state_dict.keys():
                new_dict[key.replace("module.", "")] = state_dict[key]
            state_dict = new_dict
            
            model = model.model
            del model.dropout
            del model.classifier
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        else:
            missing_keys, unexpected_keys = load_model(model.model, cfg.get('pretrained_checkpoint'), strict=False)
            model = model.model

        #################################

        print("-------------------------------------------------")
        print("Pretrained Checkpoint: " + str(checkpoint))
        print("-------------------------------------------------")

        if len(missing_keys) > 0:
            print(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        if len(missing_keys) > 5:
            print("Critical error! Missing keys: " + len(missing_keys))
            assert False

        ######################################

        self.model = model.bert
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.get('pretrained_model_name'),
            model_max_length=cfg.get("evaluation_max_seq_len"))

        print("M2 BERT Encoder Loaded")

        print(model)
        print("Max Sequence Length for Model")
        print(cfg.get("max_seq_len"))
        print("Max Sequence Length for Classification")
        print(cfg.get("evaluation_max_seq_len"))
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        encoded_queries = torch.FloatTensor([]).cpu()
        for queries_chunk in tqdm(split_list(queries, batch_size), total=len(queries) // batch_size + (1 if len(queries) % batch_size != 0 else 0)):
            
            final_input_ids = None
            for query in queries_chunk:
                input_ids = self.tokenizer([query], return_tensors="pt").to(self.device)

                if input_ids['input_ids'].shape[1] < self.cfg.get("evaluation_max_seq_len"):
                    input_ids['input_ids'] = expand_tensor(input_ids['input_ids'], self.cfg)
                    input_ids['attention_mask'] = expand_tensor(input_ids['attention_mask'], self.cfg)

                input_ids['input_ids'] = input_ids['input_ids'][:, :self.cfg.get('evaluation_max_seq_len')]
                input_ids['attention_mask'] = input_ids['attention_mask'][:, :self.cfg.get('evaluation_max_seq_len')]

                if final_input_ids == None:
                    final_input_ids = input_ids
                else:
                    final_input_ids['input_ids'] = torch.cat((final_input_ids['input_ids'], input_ids['input_ids']), 0)
                    final_input_ids['attention_mask'] = torch.cat((final_input_ids['attention_mask'], input_ids['attention_mask']), 0)

            input_ids = final_input_ids

            assert input_ids['input_ids'].shape[1] == self.cfg.get("evaluation_max_seq_len")
            assert input_ids['attention_mask'].shape[1] == self.cfg.get("evaluation_max_seq_len")

            with torch.no_grad():
                encoded_text = self.model(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'])

                embedding = encoded_text[1].detach().cpu()

                encoded_queries = torch.cat((encoded_queries, embedding), 0)

        if torch.isnan(encoded_queries).any():
            print("NaNs in encoded_queries")
            raise ValueError()

        return encoded_queries.numpy()
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:

        encoded_passages = torch.FloatTensor([]).cpu()

        for passages_chunk in tqdm(split_list(corpus, batch_size), total=len(corpus) // batch_size + (1 if len(corpus) % batch_size != 0 else 0)):
            if self.cfg.get("evaluation_max_seq_len") >= 512:
                passages_chunk = [passage['title'] + " " + passage['text'] for passage in passages_chunk]
            else:
                passages_chunk = [passage['text'] for passage in passages_chunk]
            
            final_input_ids = None
            for passage in passages_chunk:
                input_ids = self.tokenizer([passage], return_tensors="pt").to(self.device)

                if input_ids['input_ids'].shape[1] < self.cfg.get("evaluation_max_seq_len"):
                    input_ids['input_ids'] = expand_tensor(input_ids['input_ids'], self.cfg)
                    input_ids['attention_mask'] = expand_tensor(input_ids['attention_mask'], self.cfg)

                input_ids['input_ids'] = input_ids['input_ids'][:, :self.cfg.get('evaluation_max_seq_len')]
                input_ids['attention_mask'] = input_ids['attention_mask'][:, :self.cfg.get('evaluation_max_seq_len')]

                if final_input_ids == None:
                    final_input_ids = input_ids
                else:
                    final_input_ids['input_ids'] = torch.cat((final_input_ids['input_ids'], input_ids['input_ids']), 0)
                    final_input_ids['attention_mask'] = torch.cat((final_input_ids['attention_mask'], input_ids['attention_mask']), 0)

            input_ids = final_input_ids

            assert input_ids['input_ids'].shape[1] == self.cfg.get("evaluation_max_seq_len")
            assert input_ids['attention_mask'].shape[1] == self.cfg.get("evaluation_max_seq_len")
            
            with torch.no_grad():
                encoded_text = self.model(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'])

                embedding = encoded_text[1].detach().cpu()

                encoded_passages = torch.cat((encoded_passages, embedding), 0)

        if torch.isnan(encoded_passages).any():
            print("NaNs in encoded_passages")
            raise ValueError()

        return encoded_passages.numpy()
    
#################################################################
    
def generate_together_embeddings(text: str, model_api_string: str, api_key: str):
    url = "https://api.together.xyz/api/v1/embeddings"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    session = requests.Session()
    response = session.post(
        url,
        headers=headers,
        json={
            "input": text,
            "model": model_api_string
        }
    )
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    return response.json()['data'][0]['embedding']
    
class Together_Encoder:
    def __init__(self, cfg, api_key, together_model_name, **kwargs):
        self.cfg = cfg
        self.api_key = api_key
        self.together_model_name = together_model_name
        self.encoder_batch_size = 1

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        
        total_encoded_queries = []
        for query_chunks in tqdm(split_list(queries, self.encoder_batch_size), total = len(queries) // self.encoder_batch_size + (1 if len(queries) % self.encoder_batch_size != 0 else 0)):
            try:
                encoded_queries = generate_together_embeddings(text=query_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            except:
                time.sleep(30)
                encoded_queries = generate_together_embeddings(text=query_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            assert type(encoded_queries) == list
            assert len(encoded_queries) == 768
            # breakpoint()
            total_encoded_queries.append(encoded_queries)
        return np.array(total_encoded_queries)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        
        passages = [passage['title'] + " " + passage['text'] for passage in corpus]
        
        total_encoded_passages = []
        for passage_chunks in tqdm(split_list(passages, self.encoder_batch_size), total = len(passages) // self.encoder_batch_size + (1 if len(passages) % self.encoder_batch_size != 0 else 0)):
            # breakpoint()
            try:
                encoded_passages = generate_together_embeddings(text=passage_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            except:
                time.sleep(30)
                encoded_passages = generate_together_embeddings(text=passage_chunks[0], model_api_string=self.together_model_name, api_key=self.api_key)
            assert type(encoded_passages) == list
            assert len(encoded_passages) == 768
            total_encoded_passages.append(encoded_passages)
        return np.array(total_encoded_passages)
