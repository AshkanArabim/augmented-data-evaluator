# Augmented Data Evaluator
Two scripts that evaluate the performance of a KNN model (paper documenting the KNN model's architecture will be added once it's published) given normal or augmented datasets.

**NOTE**: The `sourcefiles` directory is sourced from [this repo](https://github.com/andysegura89/Pragmatic_Similarity_ISG) by Andres Segura. Some modifications were made (e.g. re-implementing KNN in PyTorch, support for two datasets, etc.) to add minor features / performance optimizations.

## Usage
Evaluation works by iterating over all participants, using them as the test subject (to be classified) based on all other participants (training data). Results are saved in the `results/` directory

- `baseline_performance.py`: Takes one dataset that is used for both training and testing. 
- `augmented_training_normal_testing_performance.py`: Takes two datasets, one augmented, and one original. It uses the augmented dataset for training, and the original dataset for testing.

## TODO
- Detailed docs, covering proper usage of the scripts
- Guidelines oh datasets' folder structure
- General cleanup
