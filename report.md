# Report: Vision Transformer Mini-System on CIFAR-10



## 1. Patch embedding

The notebook converts each image into a token sequence with a convolutional patch projection:

- kernel size = patch size
- stride = patch size
- output channel count = token dimension

The token dimension is fixed to `128`.

Final tested patch sizes:

- `4`
- `8`

Sequence lengths:

- `patch4_cls`: `8 x 8 + 1 = 65`
- `patch4_mean`: `8 x 8 = 64`
- `patch8_cls`: `4 x 4 + 1 = 17`
- `patch8_mean`: `4 x 4 = 16`

This directly shows the key ViT tradeoff: smaller patches preserve more spatial detail, but they increase sequence length and therefore attention cost.

## 2. Implemented  ViT

The notebook implements a  Vision Transformer classifier with:

- patch embedding from image tensor to token sequence
- learned positional embeddings
- two readout strategies:
  - class token
  - mean pooling over tokens
- `4` self-attention encoder blocks
- a linear classification head for `10` classes

The tensor flow is:

1. image -> patch tokens
2. add positional embeddings
3. optionally prepend a class token
4. pass tokens through `4` transformer blocks
5. read out either the class token or the mean-pooled token representation
6. apply the classification head

## 3. Final configuration comparison

The final notebook compares exactly four configurations:

The attention memory and FLOPs columns are approximate analytical calculations, not exact profiler measurements. They are included to make the self-attention cost comparable across configurations.

| Name | Patch | Pooling | Params | Seq len | Attention memory / block | Attention FLOPs / block | Forward time proxy | Val acc | Test acc |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch4_mean` | 4 | mean | 809,098 | 64 | 8.000 MB | 268.435 M | 10.309 ms | 0.6592 | 0.6603 |
| `patch4_cls` | 4 | cls | 809,354 | 65 | 8.252 MB | 276.890 M | 10.449 ms | 0.6366 | 0.6411 |
| `patch8_mean` | 8 | mean | 821,386 | 16 | 0.500 MB | 16.777 M | 3.170 ms | 0.6158 | 0.6080 |
| `patch8_cls` | 8 | cls | 821,642 | 17 | 0.564 MB | 18.940 M | 2.914 ms | 0.5816 | 0.5832 |


Parameter count changes only a little, while sequence length changes dramatically. The main computation tradeoff is therefore driven by token count, not by parameter count.

## 4. What patch size changed

Patch size had the strongest effect in the whole study.

- best patch-size-4 model: `patch4_mean`, test accuracy `0.6603`
- best patch-size-8 model: `patch8_mean`, test accuracy `0.6080`

Why:

- patch size `4` keeps more local detail inside the token sequence
- CIFAR-10 classes such as `cat`, `dog`, `bird`, `deer`, `truck`, and `automobile` often differ by small shape and texture cues
- patch size `8` is cheaper, but it discards too much fine structure too early

The cost increase is large:

- `patch4_mean` attention memory per block: `8.000 MB`
- `patch8_mean` attention memory per block: `0.500 MB`

So the better model is also roughly `16x` heavier in attention-score memory.

## 5. What pooling choice changed

Pooling choice mattered, but much less than patch size.

Mean pooling beat the class token at both patch sizes:

- patch size `4`: `patch4_mean` vs `patch4_cls`
  - validation: `0.6592` vs `0.6366`
  - test: `0.6603` vs `0.6411`
- patch size `8`: `patch8_mean` vs `patch8_cls`
  - validation: `0.6158` vs `0.5816`
  - test: `0.6080` vs `0.5832`

Interpretation:

- in this small ViT, mean pooling is a simpler and more effective readout
- the class token adds one more token and a little more cost, but it did not improve the representation enough to help accuracy

## 6. Error analysis

### Hardest classes

The lowest per-class accuracies of the best model are:

| Class | Accuracy |
| --- | ---: |
| cat | 0.352 |
| deer | 0.491 |
| dog | 0.592 |
| bird | 0.663 |
| truck | 0.682 |

`cat` is clearly the hardest class.

### Strongest confusions

The most common confusion pairs are:

| True class | Predicted class | Count |
| --- | --- | ---: |
| deer | bird | 228 |
| cat | dog | 221 |
| truck | automobile | 184 |
| cat | bird | 175 |
| deer | horse | 142 |
| ship | airplane | 140 |
| frog | bird | 133 |
| dog | cat | 127 |

### Why these errors happen

The hardest images in the saved misclassification panel follow the same pattern as the confusion table:

- small animals with similar silhouettes and textures are difficult at `32 x 32`
- `truck` and `automobile` often differ by details that become weak after aggressive tokenization
- side-view `ship` and `airplane` examples can share coarse elongated shapes in low resolution

This is exactly where smaller patches help: they preserve more local evidence before self-attention mixes the tokens.

## 7. Architecture

### Architectural Discussion

An flat token-based vision pipeline is most practical when the target is to model an image as a regular sequence of patches, and investigate Transformer mechanics directly. In that context, patch embedding, positional encoding, sequence length and self-attention cost are all explicit things that lend themselves well to comparing configurations. 

This is useful, but becomes limiting if the visual detail is highly local (e.g. texture details) or multi-scale (e.g. image stitching). When one uses large patches, it loses fine structure too early, and in contrast when one uses small ones then it preserves more detail at the cost of significantly  increasing the cost of attention because its cost grows quadratically with sequence length. As suggested by this comparison, Flat ViT-style pipelines also lack a strong built-in locality bias compared to convolutional models, meaning they may be less data-efficient or simply less natural for tasks where the local spatial structure is especially important.

It is this homework that makes those trade-offs visible. While recognition quality can be superior when smaller patches are used, since the model trains with a finer tokenization of the image, this increases sequence length (and consequently attention) and is high cost. This is precisely the virtue of a flat token pipeline: it is simple, clean and easy to analyze, but its computational cost and weak locality prior become increasingly obvious as the visual target also becomes more detailed or more scale-sensitive.




