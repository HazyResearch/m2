

from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import os
import argparse

from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import pdb
import torch
from tabulate import tabulate

from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from typing import Optional, cast
from datetime import datetime

import torch
import torch.nn.functional as F

from src.embeddings.create_loco import load_loco_from_hf
from src.embeddings.dres import DenseRetrievalExactSearch as DRES

from embeddings_inference import OpenAI_Encoder, Voyager_Encoder, Cohere_Encoder, M2_BERT_Encoder, Together_Encoder

import time
from sentence_transformers import SentenceTransformer

######################################################################

from beir import util, LoggingHandler
import pathlib, os
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
import logging

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

######################################################################

import argparse

parser = argparse.ArgumentParser(description='Your program description here.')

model_options = [
    'm2',
    'sentence-bert',
    'openai',
    'voyager',
    'cohere'
]

'''
suggested model names:
sentence_bert_model: "BAAI/bge-large-en-v1.5"
openai_embedding_model: "text-embedding-ada-002"
voyager_embedding_model: "voyage-01"
cohere_embedding_model: "embed-english-v3.0"
'''

# Boolean flags
parser.add_argument('--model', type=str, default='m2', choices=model_options, help='Model type')
parser.add_argument('--model-name', type=str, default="togethercomputer/m2-bert-80M-32k-retrieval", help='Model name')

parser.add_argument('--together-api', action='store_true', help='Use Together API')
parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings')
parser.add_argument('--save_embeddings_prefix', type=str, default="embeddings")

# File paths
parser.add_argument('--yaml-file', type=str, default="yamls/embeddings/m2-bert-80M-32k-retrieval.yaml", help='Path to YAML file')
parser.add_argument('--checkpoint', type=str, help='M2 pretrained checkpoint')

# Integer argument
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for encoding')

# Baselines
parser.add_argument('--perform-BM25-and-reranking-with-BGE', action='store_true', help='Perform BM25 and reranking with BGE')

args = parser.parse_args()

# Model Selection
use_M2_BERT = args.model == 'm2'
use_sentence_BERT_model = args.model == 'sentence-bert'
use_OpenAI = args.model == 'openai'
use_Voyager = args.model == 'voyager'
use_Cohere = args.model == 'cohere'

use_Together_API = args.together_api
if use_Together_API:
    try:
        TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']
    except:
        'Please set your Together API key as an environment variable called TOGETHER_API_KEY'
save_embeddings = args.save_embeddings

yaml_file = args.yaml_file
checkpoint = args.checkpoint

if use_M2_BERT and not use_Together_API and checkpoint is None:
    checkpoint = hf_hub_download(
        repo_id = args.model_name,
        filename = "pytorch_model.bin"
    )

batch_size_for_encoding = args.batch_size
perform_BM25_and_reranking_with_BGE = args.perform_BM25_and_reranking_with_BGE

tau_scrolls_summ_screen_fd_config = ("tau/scrolls", "test", "input", "output", "summ_screen_fd")
tau_scrolls_gov_report_config = ("tau/scrolls", "test", "input", "output", "gov_report")
tau_scrolls_qmsum_config = ("tau/scrolls", "test", "input", "output", "qmsum")
qasper_title_config = ("qasper", "test", "full_text", "title", None)
qasper_abstract_config = ("qasper", "test", "full_text", "abstract", None)

multifieldqa_en_config = ("long_bench", "test", "context", "input", "multifieldqa_en")
wikimqa_config = ("long_bench", "test", "context", "input", "2wikimqa")
passage_retrieval_en_config = ("long_bench", "test", "context", "input", "passage_retrieval_en")
legal_case_reports = ("legal_case_reports", "test", None, None, None)
courtlistener_html = ("courtlistener", "test", "Document_HTML", "Query", "Document_HTML")
courtlistener_plain_text = ("courtlistener", "test", "Document_Plain_Text", "Query", "Document_Plain_Text")
stackoverflow = ("stackoverflow", "test", "passage", "query", None)

total_datasets = [
    tau_scrolls_summ_screen_fd_config, tau_scrolls_gov_report_config,
    tau_scrolls_qmsum_config, qasper_title_config, qasper_abstract_config,
    multifieldqa_en_config, wikimqa_config,
    passage_retrieval_en_config, legal_case_reports,
    courtlistener_html, courtlistener_plain_text, stackoverflow
]

######################################################################

column_names = ["Dataset", "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10", "NDCG@100", "NDCG@1000"]
rows = [column_names]

with open(yaml_file) as f:
    yaml_cfg = om.load(f)
cfg = yaml_cfg

cfg = cfg.model

if use_M2_BERT and not use_Together_API:
    print("Model YAML Used")
    print(yaml_file)

######################################################################

document_statistics_columns = ['Dataset', "Query Average Length", "Document Average Length",
                               "Query Median Length", "Document Median Length",
                               "Query Min. Length", "Query Max. Length", 
                               "Document Min. Length", "Document Max. Length"]
document_statistics_rows = [document_statistics_columns]

for dataset in total_datasets:

    if type(dataset) == str:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        print("Dataset downloaded here: {}".format(data_path))

        current_row = [dataset]
        document_statistics_row = [dataset]
        print(f"Starting on {dataset}!")

        data_path = "datasets/" + dataset #data_path = "datasets/scifact"
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev"

        dataset_name = dataset

    else:
        
        current_row = [dataset[0] + "_" + str(dataset[4]) + "_" + str(dataset[3])]
        document_statistics_row = [dataset[0] + "_" + str(dataset[4]) + "_" + str(dataset[3])]
        print(f"Starting on {dataset[0]}_{dataset[4]}_{dataset[3]}!")

        dataset_name = f'{dataset[0]}_{dataset[4]}_{dataset[3]}'

        corpus, queries, qrels = load_loco_from_hf(dataset[0], dataset[1], dataset[2], dataset[3], subset=dataset[4])

    ######################################################################

    def calculate_query_document_lengths(corpus, queries):
        document_lengths = []
        query_lengths = []
        for corpus_id in corpus:
            document_lengths.append(len(corpus[corpus_id]['text']))
        for query_id in queries:
            query_lengths.append(len(queries[query_id]))
        return document_lengths, query_lengths

    import statistics
    passage_lengths, query_lengths = calculate_query_document_lengths(corpus, queries)
    document_statistics_row += [round(sum(query_lengths) / len(query_lengths), 0), round(sum(passage_lengths) / len(passage_lengths), 0)]
    document_statistics_row += [round(statistics.median(query_lengths), 2), round(statistics.median(passage_lengths), 2)]
    document_statistics_row += [min(query_lengths), max(query_lengths)]
    document_statistics_row += [min(passage_lengths), max(passage_lengths)]
    document_statistics_rows.append(document_statistics_row)

    ######################################################################

    if not perform_BM25_and_reranking_with_BGE:

        if use_OpenAI:
            print("Initializing OpenAI Encoder!")
            openai_encoder = OpenAI_Encoder(embedding_model=args.model_name)
            model = DRES(openai_encoder, batch_size=batch_size_for_encoding)
        elif use_Voyager:
            print("Initializing Voyager Encoder!")
            openai_encoder = Voyager_Encoder(embedding_model=args.model_name)
            model = DRES(openai_encoder, batch_size=batch_size_for_encoding)
        elif use_Cohere:
            print("Initializing Cohere Encoder!")
            openai_encoder = Cohere_Encoder(truncation="END", embedding_model=args.model_name)
            model = DRES(openai_encoder, batch_size=batch_size_for_encoding)
        elif not use_M2_BERT and use_sentence_BERT_model:
            model = DRES(models.SentenceBERT(args.model_name), batch_size=batch_size_for_encoding)
        else:
            if use_Together_API:
                m2_encoder = Together_Encoder(cfg=cfg, api_key=TOGETHER_API_KEY, together_model_name=args.model_name)
                model = DRES(m2_encoder, batch_size=batch_size_for_encoding)
            else:
                m2_encoder = M2_BERT_Encoder(checkpoint=checkpoint, cfg=cfg)
                model = DRES(m2_encoder, batch_size=batch_size_for_encoding)

        retriever = EvaluateRetrieval(model, score_function="cos_sim")

        #### Retrieve dense results (format of results is identical to qrels)
        results = retriever.retrieve(corpus, queries)
        if save_embeddings:
            np.save(f'embeddings/{dataset_name}_{args.save_embedding_prefix}_query.npy'.replace('/', '_'), model.query_embeddings)
            np.save(f'embeddings/{dataset_name}_{args.save_embedding_prefix}_corpus.npy'.replace('/', '_'), model.sub_corpus_embeddings)

        ######################################################################

        #### Evaluate your retrieval using NDCG@k, MAP@K ...

        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        logging.info("--------------------------------------------------------------")
        if type(dataset) == str:
            logging.info("Dataset Evaluated: " + dataset)
        else:
            logging.info("Dataset Evaluated: " + str(dataset[0]) + "_" + str(dataset[4]) + "_" + str(dataset[3]))
        logging.info("--------------------------------------------------------------")
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        print("NDCG")
        print(ndcg)
        logging.info("--------------------------------------------------------------")

        for column in column_names[1:]:
            current_row.append(ndcg[column])

        rows.append(current_row)

    else:

        from beir.retrieval.search.lexical import BM25Search as BM25
        from beir.retrieval.evaluation import EvaluateRetrieval

        #### Provide parameters for elastic-search
        hostname = "localhost" 
        index_name = dataset[0].replace("/","-") 
        initialize = True # True, will delete existing index with same name and reindex all documents

        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
        retriever = EvaluateRetrieval(model)

        #### Retrieve dense results (format of results is identical to qrels)
        results = retriever.retrieve(corpus, queries)
        retriever.delete(index_name)

        ###########################

        from beir.reranking.models import CrossEncoder
        from beir.reranking import Rerank

        #### Reranking using Cross-Encoder models (list: )
        cross_encoder_model = CrossEncoder(cross_encoder_model_choice)
        reranker = Rerank(cross_encoder_model, batch_size=8)

        # Rerank top-100 results using the reranker provided
        rerank_results = reranker.rerank(corpus, queries, results, top_k=20)
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)
        print("NDCG")
        print(ndcg)
        logging.info("--------------------------------------------------------------")

        for column in column_names[1:]:
            current_row.append(ndcg[column])

        rows.append(current_row)

######################################################################

df = pd.DataFrame(rows, columns=column_names)

######################################################################

print("------------------------------------------------")
print(tabulate(df, tablefmt="grid"))
print("------------------------------------------------")

######################################################################

