# Homework 2: Vision Transformer on CIFAR-10


## Deliverables

- `hw2_vit_cifar10.ipynb` - main standalone notebook with all model, training, evaluation, and visualization code inside notebook cells
- `report.md` - short written report that directly answers the assignment questions
- `outputs/` - final artifacts kept for submission and review
- `cifar-10-python.tar.gz` - local CIFAR-10 archive used by the notebook


## Patch embedding

The notebook implements patch embedding as a `Conv2d` projection with kernel size and stride equal to the patch size.

- image size: `32 x 32 x 3`
- token dimension: `128`
- tested patch sizes: `4` and `8`

Sequence lengths:

- `patch4_cls`: `8 x 8 + 1 = 65`
- `patch4_mean`: `8 x 8 = 64`
- `patch8_cls`: `4 x 4 + 1 = 17`
- `patch8_mean`: `4 x 4 = 16`

## Implemented ViT

The notebook implements a Vision Transformer with:

- patch embedding from image tensors to token sequences
- learned positional embeddings
- either a class token or mean pooling over patch tokens
- `4` transformer encoder blocks with multi-head self-attention
- a linear classification head for the `10` CIFAR-10 classes


## Final compared configurations

The final comparison contains exactly four configurations:

| Name | Patch size | Pooling | Depth | Token dim | Sequence length |
| --- | ---: | --- | ---: | ---: | ---: |
| `patch4_cls` | 4 | class token | 4 | 128 | 65 |
| `patch4_mean` | 4 | mean pooling | 4 | 128 | 64 |
| `patch8_cls` | 8 | class token | 4 | 128 | 17 |
| `patch8_mean` | 8 | mean pooling | 4 | 128 | 16 |

## Final results

These values come from the current executed notebook and the saved table at `outputs/comparison.csv`.

| Name | Params | Sequence length | Attention memory / block | Attention FLOPs / block | Forward time proxy | Val acc | Test acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch4_mean` | 809,098 | 64 | 8.000 MB | 268.435 M | 10.309 ms | 0.6592 | 0.6603 |
| `patch4_cls` | 809,354 | 65 | 8.252 MB | 276.890 M | 10.449 ms | 0.6366 | 0.6411 |
| `patch8_mean` | 821,386 | 16 | 0.500 MB | 16.777 M | 3.170 ms | 0.6158 | 0.6080 |
| `patch8_cls` | 821,642 | 17 | 0.564 MB | 18.940 M | 2.914 ms | 0.5816 | 0.5832 |

Direct takeaways:

- the best final model is `patch4_mean`
- reducing patch size from `8` to `4` gives the biggest accuracy gain, but attention becomes about `16x` more expensive
- mean pooling beats the class token at both patch sizes in this compact setting
- parameter count changes only slightly across variants; sequence length is the dominant driver of attention cost

## Error analysis summary

The hardest classes in the best model are:

- `cat`: `0.352`
- `deer`: `0.491`
- `dog`: `0.592`
- `bird`: `0.663`

The strongest confusion pairs are:

- `deer -> bird`: `228`
- `cat -> dog`: `221`
- `truck -> automobile`: `184`
- `cat -> bird`: `175`
- `deer -> horse`: `142`

What this means:

- smaller patches help because CIFAR-10 classes often differ by fine local details
- mean pooling was more stable than the class token for this small ViT
- the hardest examples are low-resolution animals with overlapping shapes and texture cues, plus visually similar vehicles in side-view images

## Final outputs kept in `outputs/`

Only the final review artifacts are kept:

- `comparison.csv` - main final comparison table
- `patch_grid_preview.png` - tokenization visualization
- `patch4_cls_curves.png`
- `patch4_mean_curves.png`
- `patch8_cls_curves.png`
- `patch8_mean_curves.png`
- `best_confusion_matrix.png`
- `best_misclassifications.png`
- `per_class_accuracy.csv`
- `top_confusions.csv`

## How to run

1. Create an environment and install dependencies:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Launch the notebook:

   ```bash
   jupyter notebook hw2_vit_cifar10.ipynb
   ```

3. Run the notebook from top to bottom.

## Notes

- the notebook uses the local CIFAR-10 archive and does not download the dataset from the network
