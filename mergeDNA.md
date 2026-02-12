## Problem to solve

- Genomic data varies in information density
    - Coding sequences vs "artifacts": repeats, regulatory functions, telomeres
-  Hard to define a meaningful vocabulary beyond base-level resolution
- DNA data is very large and features long range dependencies

A base-level architecture with long-range models wastes time on "boring" regions.
A learned or kmer level tokenizer fails to capture global information.

Key ideas:
- token merging gives granularity of context size
- adaptive pre-training objectives prioritise informative regions


## Core architecture

- Nested autoencoder structure
    - Local encoder with ToMe chunks bases into variable length tokens
    - Latent encoder looks for global context  on the tokens
    - Symmetric decoders to reconstruct sequences


### Local encoder

- Input X, vector of (encoded) nucleotides, in V^N, V = {0,1,2,3,4,5,6} (A,C,T,G,N,Pad,Mask)
- Output Z_L, S
- Z_L in R^{L x D} for some desired length L, embedding dimension D
- S in {0,1}^{L x N} indicates the merging i.e. S[i,j] = 1 if j merged into token i     
    - Implemented by stack of alternating self attention and merging layers
    - S implemented here by source_map in {0..L_new-1}^L_prev, source_map[j] = i for each layer
- Attention with local-window maintains linear complexity
- Since windows are small, only a few merges needed (merge r out of W tokens), logarithmic rather than linear of RNN or SSM (HyenaDNA)

T_local = O(N W D) for forward pass

### Latent encoder

- Input Z_L in R^{L x D}
- Stack of transformers with full attention 
- Optional token merging as part of the latent MTR and AMTM objectives 
    - If no ToMe, output Z'_L in R^{L x D}
    - If ToMe, output Z'_K in R^{K x D} and new source index S' (or equiv, source map)


T_latent = T_self-attention + T_feed-forward = O( num_transformers * (L^2 D + L D^2) ) ~ O(N^3 + N D^2) 

### Latent decoder / local decoder

- Symmetric versions of their corresponding encoders
- Often detached for downstream tasks


### Token merging

- Partition tokens A, B
- Match each a to its most similar b
- Keep r most similar edges
- Merge connections by adding
- Differentiable!


## Loss functions

### Merged token reconstruction (MTR)

- Cross entropy between reconstructed and original tokens
- Should sample different levels of compression (not implemented)

### Latent MTR

- MTR but with ToMe at the latent encoder level and with the local encoder held fixed
- Pushes latent encoder to produce compact representations
- Downweighted by lambda = 0.25 in practice

### Adaptive Masked Token Modeling (AMTM)

- Based on the latent encoder merging from previous step
- Mask important tokens i.e. those not heavily merged
- Upsample back to original space and mask the input
- Masked language modeling but focused on high-value token base positions


## Suitability of MergeDNA

- Local attention -> strong at local modelling
- Global modelling only over compressed tokens
- Cross-segment interactions dependent on the summary tokens
- e.g. local tokens could capture small motifs (5-20bp), and then interactions between motifs can be modelled effectively by the latent encoder

Fundamentally, DNA is multi-scale

Good tasks:
- Whole-chromosome or whole genome modelling where base-resolution models can't scale
- Variant effect prediction with large amounts of context, maybe especially for e.g. CNVs or other small SVs rather than SNPs

Bad tasks:
- Anything requiring base level predictions e.g. CRISPR where a small handful of bps mismatches can lead to off-target activity; need fine grained understanding

Tasks requiring care:
- Promoters/enhancers: effects happen at long range, so need the regions to be suitably modelled by the local tokens. e.g. does the enhancer carry enough signal in its local region to not be viewed as uninformative by the local encoder?


e.g. compare to HyenaDNA

- Implicit convolutions, nucleotide resolution, global mixing is direct
- Treats whole genome as a global signal
- Better for global pattern finding vs modular organisation of merge
- Faster (fully linear) rather than transformer based
- Merge's tokens may be more interpretable than HyenaDNA

Is Merge's inner layer the best choice? Hyena on the tokens?


## Implementation choices

- No side effects in forward
- Didn't implement Flash Attention (as mentioned in paper) - clear optimisation

Desirable test suite:
- Output shape and type of each module
- Check values of gradients in backward() for each loss
- Small training end-to-end test
- Test checkpoint, reload -> same forward outputs

Desirable performance testing:
- Other benchmark experiments
- Ablation testing:
    - e.g. sequentially using each of the 3 loss functions
    - downweighting of latent mtr
    - encoder layer config / aggressiveness of compression


