

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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

##################################################

import torch
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from typing import Optional, cast
import src.create_bert as bert_module

################################################

from src.embeddings.training_functions import gather_LoCo_training_examples, gather_MSMARCO_examples, expand_8k_model_to_32k

import torch
import torch.nn.functional as F
import gc
import argparse

################################################

random_state = 42
np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--checkpoint_save_steps", type=int, required=True)
    parser.add_argument("--loco_evaluation_set_count", type=int, required=True)
    parser.add_argument("--run_data_parallelism", type=bool, required=True)
    parser.add_argument("--dataset_choice", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--loss_choice", type=str, required=True)
    parser.add_argument("--query_cap_per_dataset", type=int, required=True)
    parser.add_argument("--negatives_per_query", type=int, required=True)

    args = parser.parse_args()

    # Instructions

    model_name = "bert-base-uncased"
    train_batch_size = args.train_batch_size #16 #32 #16 #32 #256
    max_seq_length = args.max_seq_length #2048 #8192 #32768
    num_epochs = args.num_epochs
    use_amp = False

    evaluation_steps_ratio = 0.1 # Ratio of steps before evaluation
    learning_rate = args.learning_rate #2e-5 #2e-5, 5e-6
    max_grad_norm = 1.0 #1.0 is the default
    train_dataloader_choice = "normal" # "no_duplicates"
    checkpoint_save_steps = args.checkpoint_save_steps #500
    loco_evaluation_set_count = args.loco_evaluation_set_count #2000

    use_negatives_from_same_dataset_for_MNRL = False
    use_negatives_from_same_dataset_for_multidataset_finetuning = True

    run_data_parallelism = args.run_data_parallelism

    ####

    dataset_choice = args.dataset_choice #"five_set_loco"
    assert dataset_choice in ["seven_set_loco", "reduced_loco", "kilt", "long_bench", "five_set_loco", "two_set_loco", "qasper", "qasper_title", "qasper_abstract", "four_set_loco"]

    from_pretrained_checkpoint = True

    use_memory_bank = False
    loss_choice = args.loss_choice # Options: "cosine_similarity_loss" #"online_contrastive_loss" #"triplet_loss" #"contrastive_loss" "multiple_negatives_ranking_loss"
    config_choice = 4
    triplet_loss_distance_metric = "cosine"
    margin = 0.5
    size_average = True

    ####

    use_msmarco_examples = False
    msmarco_examples_count = 100000 #100000 #225000
    use_nli_training_examples = False
    use_nli_training_examples_cutoff = 10000 #50000

    use_long_context_examples = True
    loco_example_count = 10000000 #500000 #250000
    threshold_for_negatives = -1 #-1 indicates negatives are randomly sampled, negative passages are randomly sampled from positive number upwards

    query_cap_per_dataset = args.query_cap_per_dataset #20000
    negatives_per_query = args.negatives_per_query #32 #64 # Number of negatives to add per query-positive passage pair

    use_M2_BERT = True
    yaml_file = "yamls/finetune-glue/M2_SC_V1.yaml"

    ############################################################

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=max_seq_length)

    ##################################################

    # Save path of the model
    directory_path = f"output/M2_BERT_80M_pretrained_{max_seq_length}_length/"
    model_save_path = directory_path + f"NO_AGGREGATION_STRATEGY_{use_msmarco_examples}_use_msmarco_examples_{msmarco_examples_count}_msmarco_examples_count-{use_long_context_examples}_use_long_context_examples-{use_long_context_examples}_use_long_context_examples" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

    checkpoint_path = model_save_path + "/checkpoints/"

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    ##################################################

    print("------------------------------------------------------")
    print("Run Configuration:")
    print("train_batch_size: " + str(train_batch_size))
    print("max_seq_length: " + str(max_seq_length))
    print("use_msmarco_examples: " + str(use_msmarco_examples))
    print("msmarco_examples_count: " + str(msmarco_examples_count))
    print("use_nli_training_examples: " + str(use_nli_training_examples))
    print("use_nli_training_examples_cutoff: " + str(use_nli_training_examples_cutoff))
    print("use_long_context_examples: " + str(use_long_context_examples))
    print("loco_example_count: " + str(loco_example_count))
    print("threshold_for_negatives: " + str(threshold_for_negatives))
    print("query_cap_per_dataset: " + str(query_cap_per_dataset))
    print("negatives_per_query: " + str(negatives_per_query))
    print("use_amp: " + str(use_amp))
    print("checkpoint_save_steps: " + str(checkpoint_save_steps))
    print("checkpoint_path: " + str(checkpoint_path))
    print("use_negatives_from_same_dataset_for_MNRL: " + str(use_negatives_from_same_dataset_for_MNRL))
    print("use_negatives_from_same_dataset_for_multidataset_finetuning: " + str(use_negatives_from_same_dataset_for_multidataset_finetuning))
    print("loss_choice: " + str(loss_choice))
    print("triplet_loss_distance_metric: " + str(triplet_loss_distance_metric))
    print("margin: " + str(margin))
    print("size_average: " + str(size_average))
    print("config_choice: " + str(config_choice))
    print("run_data_parallelism: " + str(run_data_parallelism))
    print("learning_rate: " + str(learning_rate))
    print("random_state: " + str(random_state))
    print("max_grad_norm: " + str(max_grad_norm))
    print("dataset_choice: " + str(dataset_choice))
    print("------------------------------------------------------")

    ##################################################

    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length) #padding=True, truncation=True
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')

    ##################################################

    def load_M2_model(yaml_file, pretrained_checkpoint=None):

        with open(yaml_file) as f:
            yaml_cfg = om.load(f)
        cfg = yaml_cfg
        cfg = cfg.model

        print("pretrained_checkpoint")
        print(cfg.get("pretrained_checkpoint"))
        print("expand_positional_embeddings")
        print(cfg.get("expand_positional_embeddings"))

        model = bert_module.create_bert_classification(#bert_module.create_bert_mlm(
                        num_labels=10, # Label value is insignificant since classifier is dropped
                        pretrained_model_name=cfg.pretrained_model_name,
                        pretrained_checkpoint=None,
                        model_config=cfg.get('model_config', None),
                        tokenizer_name=cfg.get('tokenizer_name', None),
                        gradient_checkpointing=cfg.get('gradient_checkpointing', None)).train()#.eval()

        if from_pretrained_checkpoint:
            state_dict = torch.load(cfg.get('pretrained_checkpoint'))['state']['model']
            if cfg.get('expand_positional_embeddings') == True:
                state_dict = expand_8k_model_to_32k(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load(cfg.get('pretrained_checkpoint'))#['state']['model']
            missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            print(f"\nFound these missing keys in the checkpoint: {', '.join(missing_keys)}\n")
        if len(unexpected_keys) > 0:
            print(f"\nFound these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}\n")
            if "state" in unexpected_keys or "rng" in unexpected_keys:
                raise ValueError("State dict not loaded properly!")

        del model.model.dropout
        del model.model.classifier

        return model

    ##################################################

    nli_dataset_path = 'data/AllNLI.tsv.gz'
    sts_dataset_path = 'data/stsbenchmark.tsv.gz'

    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    # Read the AllNLI.tsv.gz file and create the training dataset
    logging.info("Read AllNLI train dataset")

    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        train_data[sent1][label].add(sent2)


    train_data = {}
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'train':
                sent1 = row['sentence1'].strip()
                sent2 = row['sentence2'].strip()

                add_to_samples(sent1, sent2, row['label'])
                add_to_samples(sent2, sent1, row['label'])  #Also add the opposite


    nli_train_samples = []
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            nli_train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            nli_train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

    if not use_nli_training_examples:
        nli_train_samples = []
    else:
        nli_train_samples = nli_train_samples[:use_nli_training_examples_cutoff]

    ################################################################

    if use_msmarco_examples:
        msmarco_train_samples = gather_MSMARCO_examples(msmarco_examples_count, loss_choice)
        nli_train_samples = nli_train_samples + msmarco_train_samples
        random.Random(random_state).shuffle(nli_train_samples)

    if use_long_context_examples:
        long_context_training_examples, long_context_validation_examples, memory_bank_query_input_ids_to_negative_passages_dict = gather_LoCo_training_examples(loco_example_count, loco_evaluation_set_count, threshold_for_negatives, negatives_per_query, use_negatives_from_same_dataset_for_MNRL, tokenizer, use_memory_bank, query_cap_per_dataset, loss_choice, dataset_choice, use_negatives_from_same_dataset_for_multidataset_finetuning)
        nli_train_samples = nli_train_samples + long_context_training_examples
        print("---------------------------------------------------------------------------")
        print("First Example in long_context_training_examples")
        print(str(long_context_training_examples[0])[:1000])
        print("---------------------------------------------------------------------------")
        random.Random(random_state).shuffle(nli_train_samples)

    ################################################################

    logging.info("Train samples: {}".format(len(nli_train_samples)))

    # Special data loader that avoid duplicates within a batch
    if train_dataloader_choice == "no_duplicates":
        train_dataloader = datasets.NoDuplicatesDataLoader(nli_train_samples, 
                                                        batch_size=train_batch_size)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset=nli_train_samples,
            batch_size=train_batch_size, #32
            shuffle=True,
            #sampler=DistributedSampler(nli_train_samples),
        )

    ################################################################

    if use_M2_BERT:
        
        mosaic_bert_model = load_M2_model(yaml_file)

        mosaic_bert_model.model.train()
        
        ####### Data Parallel (DP)

        if run_data_parallelism:
            print("Available GPUs: " + str(torch.cuda.device_count()))
            device_ids = [id for id in range(torch.cuda.device_count())]
            data_parallel_model = nn.DataParallel(mosaic_bert_model.model, device_ids=device_ids)
            word_embedding_model._modules['auto_model'] = data_parallel_model
        else:
            word_embedding_model._modules['auto_model'] = mosaic_bert_model.model.to(torch.device("cuda:0"))#.bert

        #######

        word_embedding_model.tokenizer.model_max_length = max_seq_length

        print("word_embedding_model._modules['auto_model']")
        print(type(word_embedding_model._modules['auto_model']))
        print(word_embedding_model._modules['auto_model'])

        model = SentenceTransformer(modules=[word_embedding_model])

    else:

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    ################################################################

    # Training loss selection
    if loss_choice == "multiple_negatives_ranking_loss":
        if use_memory_bank:
            train_loss = losses.MultipleNegativesRankingLoss(model, 
                                                            memory_bank_query_input_ids_to_negative_passages_dict=memory_bank_query_input_ids_to_negative_passages_dict)
        else:
            train_loss = losses.MultipleNegativesRankingLoss(model)
    elif loss_choice == "contrastive_loss":
        train_loss = losses.ContrastiveLoss(model,
                                            margin=margin,
                                            size_average=size_average)
    elif loss_choice == "online_contrastive_loss":
        train_loss = losses.OnlineContrastiveLoss(model)
    elif loss_choice == "triplet_loss":
        if triplet_loss_distance_metric == "euclidean":
            train_loss = losses.TripletLoss(model,
                                            distance_metric=losses.TripletDistanceMetric.EUCLIDEAN,
                                            triplet_margin=margin)
        elif triplet_loss_distance_metric == "cosine":
            train_loss = losses.TripletLoss(model, 
                                            distance_metric=losses.TripletDistanceMetric.COSINE,
                                            triplet_margin=margin)
        else:
            raise ValueError("Triplet loss distance metric not found")
    elif loss_choice == "cosine_similarity_loss":
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss_choice == "mega_batch_margin_loss":
        raise ValueError("Need to reconfigure MegaBatchMarginLoss!")
        train_loss = losses.MegaBatchMarginLoss(model,
                                                use_mini_batched_version=True,
                                                mini_batch_size=4)
    else:
        raise ValueError("Loss function not found!")

    ######################################################

    if not use_long_context_examples:
        #Read STSbenchmark dataset and use it as development set
        logging.info("Read STSbenchmark dev dataset")
        dev_samples = []
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'dev':
                    score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                    dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        dev_samples = dev_samples[:loco_evaluation_set_count]
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

    else:

        print("long_context_validation_examples")
        print(len(long_context_validation_examples))
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(long_context_validation_examples, batch_size=train_batch_size, name='loco_validation')

    ######################################################

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    logging.info("Beginning fine-tuning!")

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            evaluation_steps=int(len(train_dataloader)*evaluation_steps_ratio),
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            checkpoint_save_steps=checkpoint_save_steps,
            checkpoint_path=checkpoint_path,
            optimizer_params={'lr': learning_rate},
            max_grad_norm=max_grad_norm,
            use_amp=use_amp          #Set to True, if your GPU supports FP16 operations
            )



    ##############################################################################
    #
    # Load the stored model and evaluate its performance on validation set
    #
    ##############################################################################

    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    hf_model_path = None
    if use_M2_BERT:

        hf_model_path = model_save_path + "_HF_Model_v2.pt"
        try:
            torch.save(model[0]._modules['auto_model']._modules['module'].state_dict(), hf_model_path)
        except:
            torch.save(model[0]._modules['auto_model'].state_dict(), hf_model_path)

        print("Saved M2-BERT model: " + hf_model_path)
        
        print("Not loading sentence transformer, using trained model")

    else:
        model = SentenceTransformer(model_save_path)

    ##################################################

    print("------------------------------------------------------")
    print("Run Configuration:")
    print("train_batch_size: " + str(train_batch_size))
    print("max_seq_length: " + str(max_seq_length))
    print("use_msmarco_examples: " + str(use_msmarco_examples))
    print("msmarco_examples_count: " + str(msmarco_examples_count))
    print("use_nli_training_examples: " + str(use_nli_training_examples))
    print("use_nli_training_examples_cutoff: " + str(use_nli_training_examples_cutoff))
    print("use_long_context_examples: " + str(use_long_context_examples))
    print("loco_example_count: " + str(loco_example_count))
    print("threshold_for_negatives: " + str(threshold_for_negatives))
    print("query_cap_per_dataset: " + str(query_cap_per_dataset))
    print("negatives_per_query: " + str(negatives_per_query))
    print("use_amp: " + str(use_amp))
    print("checkpoint_save_steps: " + str(checkpoint_save_steps))
    print("checkpoint_path: " + str(checkpoint_path))
    if hf_model_path is not None:
        print("hf_model_path: " + str(hf_model_path))
    print("------------------------------------------------------")

    ##################################################

    if not use_long_context_examples:
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
    else:
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(long_context_validation_examples, batch_size=train_batch_size, name='loco-validation')

    test_evaluator(model, output_path=model_save_path)

    print("Saved model to: " + model_save_path)