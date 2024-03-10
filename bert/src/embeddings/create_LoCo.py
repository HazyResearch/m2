

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
import random
import json
import os
from bs4 import BeautifulSoup

################################################################################

def load_loco_from_hf(dataset_name: str, split: str, document_column: str, query_column: str, subset=None):

    if "qasper" == dataset_name:
        if "abstract" == query_column:
            dataset_choice = "qasper_abstract"
            split = "test"
        elif "title" == query_column:
            dataset_choice = "qasper_title"
            split = "test"
        else:
            raise ValueError("No dataset specified for QASPER!")
    elif type(subset) == str and "passage_retrieval" in subset:
        dataset_choice = "passage_retrieval"
        split = "test"
    elif "legal_case_reports" == dataset_name:
        dataset_choice = dataset_name
        split = "test"
    elif "courtlistener" == dataset_name:
        if "Document_HTML" == subset:
            dataset_choice = "courtlistener_HTML"
            split = "test"
        elif "Document_Plain_Text" == subset:
            dataset_choice = "courtlistener_Plain_Text"
            split = "test"
    elif "stackoverflow" == dataset_name:
        dataset_choice = dataset_name
        split = "test"
    elif "multifieldqa_en" == subset:
        dataset_choice = "multifieldqa"
        split = "test"
    else:
        dataset_choice = subset
        split = "test"
        #raise ValueError("No dataset specified for LoCo!")


    ##########################################

    if split == "validation":
        split = "test"

    queries_dataset = load_dataset("hazyresearch/LoCoV1-Queries")[split]
    def filter_condition(example):
        return example['dataset'] == dataset_choice
    queries_dataset = queries_dataset.filter(filter_condition)

    documents = load_dataset("hazyresearch/LoCoV1-Documents")[split]
    def filter_condition(example):
        return example['dataset'] == dataset_choice
    documents = documents.filter(filter_condition)

    ##########################################

    queries = {}
    qrels = {}
    for row in tqdm(range(len(queries_dataset))):
        queries[queries_dataset[row]["qid"]] = queries_dataset[row]["query"]
        qrels_list = {}
        assert type(queries_dataset[row]["answer_pids"]) == list
        for pid in queries_dataset[row]["answer_pids"]:
            qrels_list[pid] = 1
        qrels[queries_dataset[row]["qid"]] = qrels_list
    
    corpus = {}
    for row in tqdm(range(len(documents))):
        corpus[documents[row]['pid']] = {"title": "", "text": documents[row]["passage"]}

    if "qasper" in dataset_choice:
        queries = {key: value for key, value in queries.items() if corpus[key.replace("Query", "Passage")]['text'] is not None} # Check to make sure corpus passage is not None
        corpus = {key: value for key, value in corpus.items() if corpus[key.replace("Query", "Passage")]['text'] is not None} # Check to make sure corpus passage is not None

    print("Example Query")
    print(list(queries.values())[5])
    print("Example Passage (cutoff at 200 characters)")
    print(list(corpus.values())[5]['text'][:200])

    return corpus, queries, qrels

###############################################
