# Monarch Mixer

![M2 logo](assets/m2-bert-logo.png)

**Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture**\
Daniel Y. Fu, Simran Arora*, Sabri Eyuboglu*, Jessica Grogan*, Isys Johnson*, Armin W. Thomas*, Benjamin F. Spector, Michael Poli, Atri Rudra, and Christopher Ré.\
[M2-BERT blog post](https://hazyresearch.stanford.edu/blog/2023-07-25-m2-bert) | M2-BERT checkpoints: [80M](https://huggingface.co/danfu09/m2-bert-80M) [110M](https://huggingface.co/danfu09/m2-bert-110M) | [arXiv](https://arxiv.org/abs/2310.12109)

Transformers have taken the world by a storm! The architecture is composed of two core operations: Attention for mixing information across the input sequence and MLPs for mixing information across the model dimension. Each operator scales quadratically -- the complexity of Attention is quadratic in sequence length and the complexity of an MLP is quadratic in model dimension. Ideally, we can have alternatives that scale more efficiently, while preserving Transformer-level quality. Towards this goal, we've been developing Monarch Mixer (M2), a framework for training models that are sub-quadratic in **both** sequence length and model dimension. 

![M2 diagram](assets/m2-diagram.png)

Our basic idea is to replace the major elements of a Transformer with Monarch matrices — which are a class of structured matrices that generalize the FFT and are sub-quadratic, hardware-efficient, and expressive. In Monarch Mixer, we use layers built up from Monarch matrices to do both mixing across the sequence (replacing the Attention operation) and mixing across the model dimension (replacing the dense MLP). This repo includes code and models for training Monarch Mixer architectures!

### Current Contents

**July 24:** We are excited to release Monarch Mixer BERT (M2-BERT), which has 25% fewer parameters/FLOPs than BERT, and matches in average quality on the GLUE benchmark. The [BERT folder](bert/) includes code for pretraining and finetuning BERT baselines and M2-BERT. We also release pretrained checkpoints at 128 sequence length for an 80M parameter BERT, which matches the average GLUE benchmark score of the BERT-base-uncased 110M parameter model, and a parameter matched M2-BERT model. 
