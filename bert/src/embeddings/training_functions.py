
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random
import pdb
import torch
import tarfile
from tqdm import tqdm
import numpy as np

##################################################

from create_LoCo import load_multi_news, load_pubmed_qa, load_tau_fs, load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum 
from create_LoCo import load_qasper, load_trivia_qa, load_kilt_dataset, load_long_bench, load_tau_scrolls_needle

import torch
import torch.nn.functional as F
import gc
from rank_bm25 import BM25Okapi

##################################################

def generate_hash_for_text(text: str, tokenizer):
    starting_tensor = tokenizer(text, return_tensors="pt")
    hash = sum(starting_tensor['input_ids'].tolist()[0][:1000]) # Hashed
    return hash

##################################################

def gather_strong_negatives(query, relevant_documents, bm25_index, document_set, threshold_for_negatives):
    
    top_documents = bm25_index.get_top_n(query.split(), document_set, n=threshold_for_negatives+10000)
    strong_negatives = [doc for doc in top_documents if doc not in relevant_documents]
    strong_negatives = strong_negatives[:threshold_for_negatives] # Cutoff for negatives

    for relevant_doc in relevant_documents:
        assert relevant_doc in top_documents

    negative_selected = random.choice(strong_negatives)
    assert type(negative_selected) == str
    return negative_selected


##################################################

def collect_dataset(dataset):
    corpus = []
    queries = []
    qrels = []
    if dataset[0] == "tau/multi_news": 
        corpus, queries, qrels = load_multi_news(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "pubmed_qa":
        corpus, queries, qrels = load_pubmed_qa(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "tau/fs":
        corpus, queries, qrels = load_tau_fs(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "tau/scrolls" and dataset[4] == "summ_screen_fd":
        corpus, queries, qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "tau/scrolls" and dataset[4] == "gov_report":
        corpus, queries, qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "tau/scrolls" and dataset[4] == "qmsum":
        corpus, queries, qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "qasper":
        corpus, queries, qrels = load_qasper(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "tau/mrqa" and dataset[4] == "triviaqa":
        corpus, queries, qrels = load_trivia_qa(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "kilt":
        corpus, queries, qrels = load_kilt_dataset(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "long_bench":
        corpus, queries, qrels = load_long_bench(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4])
    else:
        print("LoCo Dataset not found!")
        assert False

    print("-----------------------------------------------")
    print("Dataset: " + str(dataset[4]))
    query_lengths = [len(query) for query in list(queries.values())]
    print("Query Lengths - 25, 50, 75, and 100 Percentiles:")
    print(np.percentile(query_lengths, [25, 50, 75, 100]))
    doc_lengths = [len(doc['text']) for doc in list(corpus.values())]
    print("Document Lengths - 25, 50, 75, and 100 Percentiles:")
    print(np.percentile(doc_lengths, [25, 50, 75, 100]))
    print("-----------------------------------------------")
    
    return corpus, queries, qrels

##################################################

def gather_LoCo_training_examples(loco_example_count, loco_evaluation_set_count, threshold_for_negatives, negatives_per_query, use_negatives_from_same_dataset_for_MNRL, 
                                  tokenizer, use_memory_bank, query_cap_per_dataset, loss_choice, dataset_choice, use_negatives_from_same_dataset_for_multidataset_finetuning):

    pubmed_qa_config = ("pubmed_qa", "train", "context", "question", "pqa_labeled") #pqa_artificial

    triviaqa_config = ("tau/mrqa", "train", "context", "question", "triviaqa")
    nq_config = ("kilt", "train", "Document", "Query", "nq")
    hotpotqa_config = ("kilt", "train", "Document", "Query", "hotpotqa")
    wow_config = ("kilt", "train", "Document", "Query", "wow")
    fever_config = ("kilt", "train", "Document", "Query", "fever")

    multinews_config = ("tau/multi_news", "train", "document", "summary", None)
    tau_fs_config = ("tau/fs", "train", "article", "abstract", "arxiv")
    tau_scrolls_summ_screen_fd_config = ("tau/scrolls", "train", "input", "output", "summ_screen_fd")
    tau_scrolls_gov_report_config = ("tau/scrolls", "train", "input", "output", "gov_report")
    tau_scrolls_qmsum_config = ("tau/scrolls", "train", "input", "output", "qmsum")
    qasper_title_config = ("qasper", "train", "full_text", "title", None)
    qasper_abstract_config = ("qasper", "train", "full_text", "abstract", None)

    multifieldqa_en_config = ("long_bench", "test", "context", "input", "multifieldqa_en")
    hotpotqa_long_bench_config = ("long_bench", "test", "context", "input", "hotpotqa")
    wikimqa_config = ("long_bench", "test", "context", "input", "2wikimqa")
    gov_report_long_bench_config = ("long_bench", "test", "context", "input", "gov_report")
    multi_news_long_bench_config = ("long_bench", "test", "context", "input", "multi_news")
    passage_retrieval_en_config = ("long_bench", "test", "context", "input", "passage_retrieval_en")
    vlsp_config = ("long_bench", "test", "input", "output", "vlsp")

    if dataset_choice == "five_set_loco":
        training_datasets = [tau_scrolls_summ_screen_fd_config, tau_scrolls_gov_report_config,
                             tau_scrolls_qmsum_config, qasper_title_config, qasper_abstract_config]
    elif dataset_choice == "four_set_loco":
        training_datasets = [tau_scrolls_summ_screen_fd_config, tau_scrolls_qmsum_config, 
                             qasper_title_config, qasper_abstract_config]
    elif dataset_choice == "kilt":
        training_datasets = [nq_config, hotpotqa_config, wow_config, fever_config]
    elif dataset_choice == "kilt_needle_in_haystack":
        training_datasets = [nq_config]
    elif dataset_choice == "long_bench":
        training_datasets = [multifieldqa_en_config, hotpotqa_long_bench_config, wikimqa_config, 
                             gov_report_long_bench_config, multi_news_long_bench_config, passage_retrieval_en_config, vlsp_config]
    elif dataset_choice == "reduced_loco":
        training_datasets = [tau_scrolls_qmsum_config]
    elif dataset_choice == "two_set_loco":
        training_datasets =  [tau_scrolls_summ_screen_fd_config, tau_scrolls_gov_report_config]
    elif dataset_choice == "qasper":
        training_datasets =  [qasper_title_config, qasper_abstract_config]
    elif dataset_choice == "qasper_title":
        training_datasets =  [qasper_title_config]
    elif dataset_choice == "qasper_abstract":
        training_datasets =  [qasper_abstract_config]
    elif dataset_choice == "seven_set_loco":
        training_datasets = [multinews_config, tau_fs_config, tau_scrolls_summ_screen_fd_config, tau_scrolls_gov_report_config,
                             tau_scrolls_qmsum_config, qasper_title_config, qasper_abstract_config]
    else:
        raise ValueError("No training set selected!")

    ############################################################

    memory_bank_query_input_ids_to_negative_passages_dict = {}

    long_context_training_examples = []
    previously_used_queries = set()

    for dataset in training_datasets:

        print(f"Collecting training examples from: {dataset[0]}_{dataset[4]}_{dataset[3]}!")

        ############################################################

        # Create set of negatives across all passages
        if not use_negatives_from_same_dataset_for_multidataset_finetuning:
            total_corpus_passages = []
            for training_dataset in training_datasets:
                if training_dataset != dataset:
                    corpus, queries, qrels = collect_dataset(training_dataset)
                    for key in corpus.keys():
                        total_corpus_passages.append(corpus[key]['text'])

        ############################################################

        corpus, queries, qrels = collect_dataset(dataset)
        
        total_corpus_keys = [key for key in corpus]
        count = 0

        document_set = [corpus[key]['text'] for key in total_corpus_keys]
        tokenized_documents = [doc.split() for doc in document_set]
        bm25_index = BM25Okapi(tokenized_documents)

        progress_bar_limit = min(len(queries), query_cap_per_dataset)
        progress_bar = tqdm(range(progress_bar_limit))

        for query_key in queries:

            if count < query_cap_per_dataset:

                query = queries[query_key]
                positive_passage_keys = [key for key in qrels[query_key]]
                assert type(query) == str

                used_negative_keys = []
                for negative_count in range(negatives_per_query + 1):
                    for qrel_key in qrels[query_key]:

                        positive_passage = corpus[qrel_key]['text']
                        assert type(positive_passage) == str
                        
                        #for val in range(2): # query --> passage and passage --> query
                        for val in range(1): # query --> passage
                            
                            if random.choice([0, 1]) == 1 and not use_negatives_from_same_dataset_for_multidataset_finetuning:
                                negative_passage = random.choice(total_corpus_passages)
                            else:
                                if threshold_for_negatives < 0:
                                    random_negative_passage_key = random.choice(total_corpus_keys)
                                    while random_negative_passage_key in positive_passage_keys or random_negative_passage_key in used_negative_keys:
                                        random_negative_passage_key = random.choice(total_corpus_keys)
                                    negative_passage = corpus[random_negative_passage_key]['text']
                                    used_negative_keys.append(random_negative_passage_key)
                                    assert type(negative_passage) == str
                                else:
                                    relevant_documents = [corpus[key]['text'] for key in positive_passage_keys]
                                    negative_passage = gather_strong_negatives(query, relevant_documents, bm25_index, document_set, threshold_for_negatives)
                            
                            ######################################

                            if val % 2 == 0:
                                text_1 = query
                                text_2 = positive_passage
                            else:
                                text_1 = positive_passage
                                text_2 = query

                            if negative_count < negatives_per_query:
                                if loss_choice in ["multiple_negatives_ranking_loss", "triplet_loss", "assisted_embedding_loss"]:
                                    long_context_training_examples.append(InputExample(texts=[text_1, text_2, negative_passage]))
                                elif loss_choice in ["contrastive_loss", "online_contrastive_loss"]:
                                    long_context_training_examples.append(InputExample(texts=[text_1, text_2], label=1))
                                    long_context_training_examples.append(InputExample(texts=[text_1, negative_passage], label=0))
                                elif loss_choice in ["cosine_similarity_loss"]:
                                    long_context_training_examples.append(InputExample(texts=[text_1, text_2], label=1.0))
                                    #long_context_training_examples.append(InputExample(texts=[text_2, text_1], label=1.0))
                                    long_context_training_examples.append(InputExample(texts=[text_1, negative_passage], label=0.0))
                                elif loss_choice in ["mega_batch_margin_loss"]:
                                    long_context_training_examples.append(InputExample(texts=[text_1, text_2]))
                                else:
                                    raise ValueError("No loss function selected!")
                            else:

                                if use_memory_bank and text_1 not in previously_used_queries:

                                    negative_passages = []
                                    for _ in range(8):
                                        if threshold_for_negatives < 0:
                                            random_negative_passage_key = random.choice(total_corpus_keys)
                                            while random_negative_passage_key in positive_passage_keys or random_negative_passage_key in used_negative_keys:
                                                random_negative_passage_key = random.choice(total_corpus_keys)
                                            negative_passage = corpus[random_negative_passage_key]['text']
                                            used_negative_keys.append(random_negative_passage_key)
                                            assert type(negative_passage) == str
                                            negative_passages.append(negative_passage)
                                        else:
                                            relevant_documents = [corpus[key]['text'] for key in positive_passage_keys]
                                            negative_passage = gather_strong_negatives(query, relevant_documents, bm25_index, threshold_for_negatives)

                                    ######################################

                                    input_ids_hash = generate_hash_for_text([text_1], tokenizer)
                                    assert input_ids_hash not in memory_bank_query_input_ids_to_negative_passages_dict
                                    previously_used_queries.add(text_1)
                                    memory_bank_query_input_ids_to_negative_passages_dict[input_ids_hash] = negative_passages


            count += 1
            progress_bar.update(1)

    ##################################################

    print("Completed creating training set and memory bank!")
    if use_negatives_from_same_dataset_for_MNRL: # Increase chunk size to use negatives from the same dataset
        chunk_size = 16
    else:
        chunk_size = 1

    chunks = [long_context_training_examples[i:i + chunk_size] for i in range(0, len(long_context_training_examples), chunk_size)]
    long_context_training_examples = [item for chunk in chunks for item in chunk]

    ##################################################

    print("Total LoCo Training Examples Before Cutoff: " + str(len(long_context_training_examples)))
    long_context_training_examples = long_context_training_examples[:loco_example_count]

    print("Total Training Examples Added from LoCo: " + str(len(long_context_training_examples)))

    ##################################################

    validation_datasets = []
    for dataset in training_datasets:
        dataset = dataset[:1] + ("validation",) + dataset[2:]
        validation_datasets.append(dataset)
        
    long_context_validation_examples = []

    for dataset in validation_datasets:

        print(f"Collecting validation examples from: {dataset[0]}_{dataset[4]}_{dataset[3]}!")

        corpus, queries, qrels = collect_dataset(dataset)
        
        total_corpus_keys = [key for key in corpus]
        count = 0
        for query_key in queries:

            if count < query_cap_per_dataset:

                query = queries[query_key]
                positive_passage_keys = [key for key in qrels[query_key]]
                assert type(query) == str

                used_negative_keys = []
                for _ in range(negatives_per_query): #Create X negatives per query-positive passage
                    for qrel_key in qrels[query_key]:

                        positive_passage = corpus[qrel_key]['text']
                        assert type(positive_passage) == str
                        
                        random_negative_passage_key = random.choice(total_corpus_keys)
                        while random_negative_passage_key in positive_passage_keys or random_negative_passage_key in used_negative_keys:
                            random_negative_passage_key = random.choice(total_corpus_keys)
                        negative_passage = corpus[random_negative_passage_key]['text']
                        used_negative_keys.append(random_negative_passage_key)
                        assert type(negative_passage) == str

                        long_context_validation_examples.append(InputExample(texts=[query, positive_passage], label=1.0))
                        long_context_validation_examples.append(InputExample(texts=[query, negative_passage], label=0.0))

            count += 1

    ##################################################

    long_context_validation_examples = long_context_validation_examples[:loco_evaluation_set_count]
    print("Total LoCo Validation Examples: " + str(len(long_context_validation_examples))) 

    return long_context_training_examples, long_context_validation_examples, memory_bank_query_input_ids_to_negative_passages_dict

    
        


##################################################

def gather_MSMARCO_examples(examples_count, loss_choice):

    # specify (query, positive sample, negative sample).
    pos_neg_ration = 4

    # Maximal number of training samples we want to use
    max_train_samples = examples_count #100000 #2e7

    ### Now we read the MS Marco dataset
    data_folder = 'msmarco-data'
    os.makedirs(data_folder, exist_ok=True)


    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage


    ### Read the train queries, store in queries dict
    queries = {}
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    
    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query



    ### Now we create our training & dev data
    train_samples = []
    dev_samples = {}

    # We use 200 random queries from the train set for evaluation during training
    # Each query has at least one relevant and up to 200 irrelevant (negative) passages
    num_dev_queries = 200
    num_max_dev_negatives = 200

    # msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
    # shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
    # We extracted in the train-eval split 500 random queries that can be used for evaluation during training
    train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
    if not os.path.exists(train_eval_filepath):
        logging.info("Download "+os.path.basename(train_eval_filepath))
        util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

    with gzip.open(train_eval_filepath, 'rt') as fIn:
        for line in fIn:
            qid, pos_id, neg_id = line.strip().split()

            if qid not in dev_samples and len(dev_samples) < num_dev_queries:
                dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}

            if qid in dev_samples:
                dev_samples[qid]['positive'].add(corpus[pos_id])

                if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                    dev_samples[qid]['negative'].add(corpus[neg_id])


    # Read our training file
    train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
    if not os.path.exists(train_filepath):
        logging.info("Download "+os.path.basename(train_filepath))
        util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)

    query_lengths = []
    passage_lengths = []
    cnt = 0
    with gzip.open(train_filepath, 'rt') as fIn:
        #for line in tqdm.tqdm(fIn, unit_scale=True):
        for line in tqdm(fIn, unit_scale=True):
            qid, pos_id, neg_id = line.strip().split()

            if qid in dev_samples:
                continue

            query = queries[qid]
            if (cnt % (pos_neg_ration+1)) == 0:
                passage = corpus[pos_id]
                label = 1
            else:
                passage = corpus[neg_id]
                label = 0

            positive_passage = corpus[pos_id]
            negative_passage = corpus[neg_id]

            if loss_choice == "cosine_similarity_loss":
                train_samples.append(InputExample(texts=[query, positive_passage], label=1.0))
                train_samples.append(InputExample(texts=[query, negative_passage], label=0.0))
            else:
                #train_samples.append(InputExample(texts=[query, passage], label=label))
                train_samples.append(InputExample(texts=[query, positive_passage, negative_passage]))
                train_samples.append(InputExample(texts=[positive_passage, query, negative_passage]))

            query_lengths.append(len(query))
            passage_lengths.append(len(passage))
            cnt += 1

            if cnt >= max_train_samples:
                break

    print("Average Query Length: " + str(sum(query_lengths) / len(query_lengths)))
    print("Max Query Length: " + str(max(query_lengths)))
    print("Average Passage Length: " + str(sum(passage_lengths) / len(passage_lengths)))
    print("Max Passage Length: " + str(max(passage_lengths)))

    return train_samples

##########################################################

def expand_8k_model_to_32k(state_dict):

    model_size = 8192 #128

    breakpoint()

    print("Expanding positional embeddings")
    original_embedding = state_dict['model.bert.embeddings.position_embeddings.weight']
            
    randomized_embeddings_list = [original_embedding]
    model_size_to_range_set = {1024: 1, 128: 15, 4096: 8, 8192: 4}
    for j in range(model_size_to_range_set[model_size]):
        randomized_embeddings_list.append(original_embedding)
    state_dict['model.bert.embeddings.position_embeddings.weight'] = torch.cat(randomized_embeddings_list, axis=0)

    assert state_dict['model.bert.embeddings.position_embeddings.weight'].shape[0] in [32768] #[512, 2048, 4096, 8192]
    assert state_dict['model.bert.embeddings.position_embeddings.weight'].shape[1] in [768] #[768, 960, 1536, 1792]

    del state_dict['model.bert.embeddings.position_ids']

    for i in range(0, 12):

        def expand_parameter(current_param):
                    
            expanded_parameter = nn.Parameter(torch.zeros(current_param.shape[0], 16 * current_param.shape[1], current_param.shape[2]))
                    
            model_size_to_range_set_for_pos_z = {1024: 3, 128: 17, 4096: 11, 8192: 5}
            for k in range(2, model_size_to_range_set_for_pos_z[model_size]):
                expanded_parameter.data[:, (k - 1) * current_param.shape[1]: (k) * current_param.shape[1], :] = current_param
                    
            return expanded_parameter

        state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.z'] = expand_parameter(state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.z'])
        assert state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.z'].shape[1] in [512, 2048, 4096, 8192, 32768]
                
        state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.z'] = expand_parameter(state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.z'])
        assert state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.z'].shape[1] in [512, 2048, 4096, 8192, 32768]

        del state_dict["model.bert.encoder.layer." + str(i) + ".attention.filter_fn.pos_emb.t"]
        if model_size != 1024:
            del state_dict["model.bert.encoder.layer." + str(i) + ".attention.filter_fn2.pos_emb.t"]

    return state_dict