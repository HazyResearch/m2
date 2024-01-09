# Minimal example to embed text using M2-BERT embedding models
import numpy as np
import os
import argparse

from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf as om

from embeddings_inference import M2_BERT_Encoder, Together_Encoder

import argparse

parser = argparse.ArgumentParser(description='Your program description here.')
parser.add_argument('--text', type=str, help='Text to embed', required=True)
parser.add_argument('--output-file', type=str, help='Output file')
parser.add_argument('--model-name', type=str, default="togethercomputer/m2-bert-80M-32k-retrieval", help='Model name')
parser.add_argument('--together-api', action='store_true', help='Use Together API')

# File paths
parser.add_argument('--yaml-file', type=str, default="yamls/embeddings/m2-bert-80M-32k-retrieval.yaml", help='Path to YAML file')
parser.add_argument('--checkpoint', type=str, help='M2 pretrained checkpoint')

args = parser.parse_args()

use_Together_API = args.together_api
if use_Together_API:
    try:
        TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']
    except:
        'Please set your Together API key as an environment variable called TOGETHER_API_KEY'

yaml_file = args.yaml_file
checkpoint = args.checkpoint

if not use_Together_API and checkpoint is None:
    checkpoint = hf_hub_download(
        repo_id = args.model_name,
        filename = "model.bin"
    )

with open(yaml_file) as f:
    yaml_cfg = om.load(f)
cfg = yaml_cfg

cfg = cfg.model

if not use_Together_API:
    print("Model YAML Used")
    print(yaml_file)

if use_Together_API:
    m2_encoder = Together_Encoder(cfg=cfg, api_key=TOGETHER_API_KEY, together_model_name=args.model_name)
else:
    m2_encoder = M2_BERT_Encoder(checkpoint=checkpoint, cfg=cfg)

emb = m2_encoder.encode_queries([args.text], 1)

print('First 10 values of the embedding')
print(emb[:, :10])
if args.output_file is not None:
    np.save(args.output_file, emb)