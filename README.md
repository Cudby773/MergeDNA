## Installation

pip install (-e) .


## Implementation notes

The main model is MergeDNAModel, which exposes 4 main components:
- LocalEncoder
- LatentEncoder
- LatentDecoder
- LocalDecoder

TokenMerge and TokenUnmerge handle tokenization.

LocalAttention and PositionalEncoding are helpers for the (token merge + attention) blocks and the transformer layers respectively.

LatentClassifier exposes a tuneable head for classifier tasks (as an example of one the experiments from the paper).


/scripts contains 3 simple scripts:
- run_models.py is for testing a tiny model version
- pretrain.py trains a small model on the multi-species-genome corpus, using the full 3-stage loss function defined in the paper
- human_enhancer_cohn_test.py tests (with or without finetuning) the latent classifier on the Human Enhancer Cohn dataset from genomic-benchmarks.
