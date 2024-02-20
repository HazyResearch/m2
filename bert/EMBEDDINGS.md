# Long-Context Retrieval with M2-BERT

<p align="center">
  <img width="40%" src="../assets/monarch_in_library.png">
</p>

We're excited to release some new long-context M2-BERT models (2k, 8k, 32k) as well as embedding versions fine-tuned for long-context retrieval!
As part of this release, we're also releasing a preview of LoCo, a new benchmark for long-context retrieval.

Check out the [blog post](https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval) for more details:

**Long-Context Retrieval Models with Monarch Mixer**\
Jon Saad-Falcon, Dan Fu, Simran Arora. Blog post, Jan 11 2024.\
[Blog post](https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval).

Paper coming soon - we're releasing the code and models first to get **your feedback** on how well these early checkpoints perform, and how else we should evaluate long-context retrieval.
**Here are some particular calls to action for feedback if you’re interested in long-context retrieval:**
* If you have long-context retrieval tasks, we would love to hear how the M2-BERT retrieval models perform in the wild!
* If you have public long-context retrieval tasks or datasets that you think would be good additions to LoCo, please let us know. We’ve only included a few retrieval tasks that have long documents, but we want to grow the benchmark to be more representative!

**Table of Contents**
* [Setup](#setup)
* [Obtaining Pretrained Checkpoints](#obtaining-pretrained-checkpoints)
* [Generating Embeddings](#generating-embeddings)
* [Running LoCo](#running-loco)
* [Training](#training)

## Setup

Follow the same setup as in the general [README](README.md).
We recommend having `flash-fft-conv` installed:

```bash
pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
pip install git+https://github.com/HazyResearch/flash-fft-conv.git
```

If you don't have it installed, you can still run the code, but you will need to set `use_flashfft` to `False` in the `yamls/embeddings` files.
Model loading will print out that you are missing some parameters, but that is fine.

For inference, you additionally need to install the dependencies in [requirements-embeddings.txt](requirements-embeddings.txt).
This is mostly `beir` and a few extra libraries to run embedding models that are only available behind existing APIs.

## Obtaining Pretrained Checkpoints

You can download pretrained checkpoints from HuggingFace:
* [M2-BERT-base-80M-2K-retrieval](https://huggingface.co/togethercomputer/m2-bert-80M-2k-retrieval)
* [M2-BERT-base-80M-8K-retrieval](https://huggingface.co/togethercomputer/m2-bert-80M-8k-retrieval)
* [M2-BERT-base-80M-32K-retrieval](https://huggingface.co/togethercomputer/m2-bert-80M-32k-retrieval)

## Generating Embeddings

You can see [embed_text.py](embed_text.py) for a minimal example of generating embeddings using the M2 BERT models.
We do not recommend using this script on its own to process many documents, since it re-loads the model from scratch every time and runs with batch size 1.

```bash
python embed_text.py --text "hello world" --model-name togethercomputer/m2-bert-80M-2k-retrieval-v2 --yaml-file yamls/embeddings/m2-bert-80M-2k-retrieval.yaml

python embed_text.py --text "hello world" --model-name togethercomputer/m2-bert-80M-8k-retrieval-v2 --yaml-file yamls/embeddings/m2-bert-80M-8k-retrieval.yaml

python embed_text.py --text "hello world" --model-name togethercomputer/m2-bert-80M-32k-retrieval-v2 --yaml-file yamls/embeddings/m2-bert-80M-32k-retrieval.yaml
```

Or using the Together API (you can find your API key [here](https://api.together.xyz/settings/api-keys)):

```bash
export TOGETHER_API_KEY={YOUR API KEY HERE}

python embed_text.py --text "hello world" --model-name togethercomputer/m2-bert-80M-2k-retrieval-v2 --together-api

python embed_text.py --text "hello world" --model-name togethercomputer/m2-bert-80M-8k-retrieval-v2 --together-api

python embed_text.py --text "hello world" --model-name togethercomputer/m2-bert-80M-32k-retrieval-v2 --together-api
```

You can use the Together API to generate embeddings from any of these models by querying directly:
```Python
import os
import requests

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

print(generate_together_embeddings(
    'Hello world',
    'togethercomputer/m2-bert-80M-32k-retrieval',
    os.environ['TOGETHER_API_KEY'])[:10])
```

## Running LoCo

You can use these commands to evaluate M2-BERT on LoCoV1 locally:

```bash
python loco_eval.py --model-name togethercomputer/m2-bert-80M-2k-retrieval-v2 --yaml-file yamls/embeddings/m2-bert-80M-2k-retrieval.yaml

python loco_eval.py --model-name togethercomputer/m2-bert-80M-8k-retrieval-v2 --yaml-file yamls/embeddings/m2-bert-80M-8k-retrieval.yaml

python loco_eval.py --model-name togethercomputer/m2-bert-80M-32k-retrieval-v2 --yaml-file yamls/embeddings/m2-bert-80M-32k-retrieval.yaml
```

Or using the Together API (you can find your API key [here](https://api.together.xyz/settings/api-keys)):

```bash
export TOGETHER_API_KEY={YOUR API KEY HERE}

python loco_eval.py --model-name togethercomputer/m2-bert-80M-2k-retrieval-v2 --together-api

python loco_eval.py --model-name togethercomputer/m2-bert-80M-8k-retrieval-v2 --together-api

python loco_eval.py --model-name togethercomputer/m2-bert-80M-32k-retrieval-v2 --together-api
```

## Training

You can use [embeddings_train.py](embeddings_train.py) to train your own M2-BERT embedding models.

Follow the [main setup instructions for M2](https://github.com/HazyResearch/m2/bert) and then install the embedding specific dependencies with:

```pip install -r requirements-embeddings.txt```

To set the maximum sequence length of M2, update the ``max_seq_length`` parameter in the ``embeddings_train.py`` command and the checkpoint chosen for the training yaml: ``training_yaml``.

To use the ``sentence_transformers`` library with M2-BERT, we have to update the Python scripts for the loss functions. We can update the loss function by installing a modified version of the ``sentence-transformers``:
```
pip uninstall sentence-transformers
git clone https://github.com/jonsaadfalcon/sentence-transformers.git
cd sentence-transformers
pip install -e .
```

To download the pretrained checkpoints for M2-BERT, use the following download links:

* [M2-BERT-128](https://storage.googleapis.com/danfu-data/checkpoints/bert/pretrained_M2-BERT_checkpoints/m2-bert-128.pt)
* [M2-BERT-2k](https://storage.googleapis.com/danfu-data/checkpoints/bert/pretrained_M2-BERT_checkpoints/m2-bert-2048.pt)
* [M2-BERT-8k](https://storage.googleapis.com/danfu-data/checkpoints/bert/pretrained_M2-BERT_checkpoints/m2-bert-8192.pt)
* [M2-BERT-32k](https://storage.googleapis.com/danfu-data/checkpoints/bert/pretrained_M2-BERT_checkpoints/m2-bert-32768.pt)

For setting a training configuration, please see the example command below:

```
python embeddings_train.py \
--train_batch_size 32 \
--mini_batch_size 4 \
--max_seq_length 2048 \
--num_epochs 1 \
--checkpoint_save_steps 1000 \
--loco_evaluation_set_count 2000 \
--run_data_parallelism False \
--dataset_choice LoCoV1 \
--learning_rate 5e-6 \
--loss_choice multiple_negatives_ranking_loss \
--query_cap_per_dataset 1000000 \
--negatives_per_query 32 \
--training_yaml yamls/finetune-glue/embedding_training.yaml
```

## Expected Warnings

During query and passage encoding following message arises from the BERT tokenizer. It does not pertain to the M2 long context models and can be ignored.  
```Token indices sequence length is longer than the specified maximum sequence length for this model... Running this sequence through the model will result 
in indexing errors'''
