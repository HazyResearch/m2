# M2-BERT

We're excited to release a preview of our first M2 models -- Monarch-Mixer-BERT.
Code in this folder is forked from Tri Dao's MLPerf implementation of BERT and MosaicML's BERT examples for support clear comparison to recently released efficient Transformer baselines.

- [Setup](#setup)
- [Getting the Data](#getting-the-data)
- [Obtaining Pretrained Checkpoints](#obtaining-pretrained-checkpoints)
- [Finetuning M2-BERT](#fine-tuning-from-pretrained-checkpoints)
- [Pretraining M2-BERT](#pretraining-bert)
- [Benchmarking](#benchmarking)

### Setup

For Docker, we recommend starting from the PyTorch Docker containers to be able to install the CUDA kernels and run the benchmarking code: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.

To install the CUDA kernels:
```
git clone https://github.com/HazyResearch/m2.git
cd csrc/flashmm
python setup.py install
```

For the pure PyTorch stack, Conda works as well:
```
conda create --name m2_bert python=3.10
conda activate m2_bert
```

To install the dependencies for training:
```bash
git clone https://github.com/HazyResearch/m2.git # if you haven't already
cd bert/
pip install -r requirements.txt
```

### Getting the Data

Our experiments use Huggingface datasets for pretraining and finetuning on GLUE.

**Pretraining Datasets**

The pretraining datasets are prepared in ```src/convert_dataset.py```. This script converts the dataset to a collection of binary `.mds` files, which will then be streamed during pretraining using [mosaicml-streaming](https://streaming.docs.mosaicml.com/en/stable/).

Our released checkpoints are pretrained on [C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4). To prepare this data locally, use the following commands:
```
# To work with a small data sample for testing
python src/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train_small val

# To prepare the full C4 dataset
python src/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train val
```

To prepare [Wikipedia](https://huggingface.co/datasets/wikipedia) and/or [Bookcorpus](https://huggingface.co/datasets/bookcorpus), use the following commands:
```
# To work with Wikipedia data
python src/convert_dataset.py --dataset wikipedia --data_subset 20220301.en --out_root ./my-copy-wikipedia --splits train

# To work with Bookscorpus data
python src/convert_dataset.py --dataset bookcorpus --out_root ./my-copy-bookcorpus --splits train
```

We provide a standalone utility script to split the above training sets for Wikipedia and Bookcorpus into training and validation splits: ```python src/utils/create_val_split.py```. Enter a path to your training data before running the script.

**Finetuning Datasets**

We evaluated the quality of our models on the standard GLUE benchmark. The data processing is handled in ```src/glue/data.py``` and the datasets are automatically downloaded from Huggingface.

### Obtaining Pretrained Checkpoints

July 24: We will release a suite of BERT models in the coming weeks. Today we are releasing the following checkpoints:
1. [M2-BERT-base (80M)](https://huggingface.co/danfu09/m2-bert-80m), which matches the average GLUE score of BERT-base with 27\% fewer parameters. Its hidden dimension is 768. *Please Note* this model was pretrained and finetuned with the legacy model configuration setting ```hyena_training_additions: True```, whereas the default option and 960 dim model are with ```hyena_training_additions: False```. Our finetuning script is set to reproduce our reported results in the blog.
2. [M2-BERT-base (110M)](https://huggingface.co/danfu09/m2-bert-110m), which is parameter matched to BERT-base with hidden dimension 960.

October 21: Now we have released two additional checkpoints, equivalent to BERT-large:
1. [M2-BERT-large (260M)](https://huggingface.co/danfu09/m2-bert-260m) has 24\% fewer parameters than BERT-large and matches in GLUE average quality. Its hidden dimension is 1536 and has 12 layers.
2. [M2-BERT-large (341M)](https://huggingface.co/danfu09/m2-bert-341m) has hidden dimension 1792 and outperforms BERT-large in GLUE.

### Fine-tuning from Pretrained Checkpoints

We include ```yaml``` files to fine-tune each of the M2-BERT checkpoints as well as the BERT-base Transformer baseline. The commands are as follows:
```
# To fine-tune from M2-BERT-base (80M)
python glue.py yamls/finetune-glue/monarch-mixer-finetune-glue-768dim-80m-parameters.yaml

# To fine-tune from M2-BERT-base (110M)
python glue.py yamls/finetune-glue/monarch-mixer-finetune-glue-960dim-parameter-matched.yaml

# To fine-tune from M2-BERT-large (260M)
python glue.py yamls/finetune-glue/monarch-mixer-large-finetune-glue-1536dim-260m-parameters.yaml

# To fine-tune from M2-BERT-large (341M)
python glue.py yamls/finetune-glue/monarch-mixer-large-finetune-glue-1792dim-341m-parameters.yaml

# To fine-tune BERT-base
python glue.py yamls/finetune-glue/hf-transformer-finetune-glue-bert-base-uncased.yaml
```

A few notes for finetuning:
1. In the ```yaml``` files, enter paths to your pretrained checkpoints on disk.
2. You can modify the ```yaml``` files if you want to save the post-finetuning checkpoints and modify the save directory.
3. Following the protocol introduced in [Izsak et al., 2021](https://arxiv.org/abs/2104.07705), we finetune for the data-poor GLUE tasks (RTE, STSB, and MRPC), starting from a checkpoint obtained after fine-tuning on the MNLI task. Once you've obtained an MNLI checkpoint, you can update the checkpoint paths in the ```yaml``` files to load this in and add the suffix ```-from-mnli``` to your experiment name to run finetuning on RTE, STSB, and MRPC.

Expected performance:
| Model | Average GLUE Score |
| :---- | :----------------------: |
| M2-BERT-base (80M)  | 79.9 |
| M2-BERT-base (110M) | 80.9 |
| M2-BERT-base (260M) | 82.2 |
| M2-BERT-base (341M) | 82.8 |

### Pretraining BERT

We include ```yaml``` files to pretrain M2-BERT as well as the BERT-base Transformer baseline. The commands are as follows:

```bash
# Pretrain M2-BERT-base (80M)
composer main.py yamls/pretrain/monarch-mixer-pretrain-786dim-80m-parameters.yaml

# Pretrain M2-BERT-base (110M)
composer main.py yamls/pretrain/monarch-mixer-pretrain-960dim-parameter-matched.yaml

# Pretrain M2-BERT-base (260M)
composer monarch-mixer-large-pretrain-1536dim-260m-parameters.yaml

# Pretrain M2-BERT-base (341M)
composer monarch-mixer-large-pretrain-1791dim-341m-parameters.yaml

# Pretrain BERT-base-uncased Transformer baseline
composer main.py yamls/pretrain/hf-transformer-pretrain-bert-base-uncased.yaml
```

A few notes for pretraining:
1. Update the ```data_local``` field in the ```yaml``` files to point to your pretraining data folder.
2. Update how frequently you want to run evaluation during pretraining, how many checkpoints to save, and where to save the checkpoints on disk.
3. By default, the `composer` tool trains in distributed mode. You can modify the launch command to ```composer -n 1``` to launch on ```1``` device instead (or another number of devices).

If training on a single node, the `composer` launcher will autodetect the number of devices.

### Benchmarking

To run throughput benchmarks of the forward pass of the model at different sequence lengths, install the `flashmm` kernel and use the following commands. Batch sizes are adjusted for a 40GB A100.

**M2-BERT-80M**
```
python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-786dim-80m-parameters.yaml max_seq_len=512 device_train_microbatch_size=32 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-786dim-80m-parameters.yaml max_seq_len=1024 device_train_microbatch_size=32 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-786dim-80m-parameters.yaml max_seq_len=2048 device_train_microbatch_size=16 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-786dim-80m-parameters.yaml max_seq_len=4096 device_train_microbatch_size=8 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-786dim-80m-parameters.yaml max_seq_len=8192 device_train_microbatch_size=4 model.model_config.use_flash_mm=True
```

**M2-BERT-110M**
```
python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-960dim-parameter-matched.yaml max_seq_len=512 device_train_microbatch_size=32 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-960dim-parameter-matched.yaml max_seq_len=1024 device_train_microbatch_size=32 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-960dim-parameter-matched.yaml max_seq_len=2048 device_train_microbatch_size=16 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-960dim-parameter-matched.yaml max_seq_len=4096 device_train_microbatch_size=8 model.model_config.use_flash_mm=True

python benchmark_fwd.py yamls/pretrain/monarch-mixer-pretrain-960dim-parameter-matched.yaml max_seq_len=8192 device_train_microbatch_size=4 model.model_config.use_flash_mm=True
```

**Transformer-110M**
```
python benchmark_fwd.py yamls/pretrain/transformer-flash-attn.yaml max_seq_len=512 device_train_microbatch_size=32

python benchmark_fwd.py yamls/pretrain/transformer-flash-attn.yaml max_seq_len=1024 device_train_microbatch_size=32

python benchmark_fwd.py yamls/pretrain/transformer-flash-attn.yaml max_seq_len=2048 device_train_microbatch_size=8

python benchmark_fwd.py yamls/pretrain/transformer-flash-attn.yaml max_seq_len=4096 device_train_microbatch_size=4

python benchmark_fwd.py yamls/pretrain/transformer-flash-attn.yaml max_seq_len=8192 device_train_microbatch_size=2
```

**HuggingFace Transformer 110M**
```
python benchmark_fwd.py yamls/pretrain/hf-transformer-pretrain-bert-base-uncased.yaml max_seq_len=512 device_train_microbatch_size=32

python benchmark_fwd.py yamls/pretrain/hf-transformer-pretrain-bert-base-uncased.yaml max_seq_len=1024 device_train_microbatch_size=16

python benchmark_fwd.py yamls/pretrain/hf-transformer-pretrain-bert-base-uncased.yaml max_seq_len=2048 device_train_microbatch_size=4

python benchmark_fwd.py yamls/pretrain/hf-transformer-pretrain-bert-base-uncased.yaml max_seq_len=4096 device_train_microbatch_size=1

# this will OOM on 40GB-A100
python benchmark_fwd.py yamls/pretrain/hf-transformer-pretrain-bert-base-uncased.yaml max_seq_len=8192 device_train_microbatch_size=1
```

### Citations, Contact, and Acknowledgements

Lookout for the M2 Arxiv coming soon!! In the meantime, we're excited to release a [blog post](https://hazyresearch.stanford.edu/blog/2023-07-25-m2-bert) on M2-BERT.

If you have questions or feedback on M2-BERT, feel free to reach out to Dan Fu (danfu@stanford.edu) and Simran Arora (simran@cs.stanford.edu).

Finally, we are extremely grateful to [Together AI](https://together.xyz/) for making this work possible!!
