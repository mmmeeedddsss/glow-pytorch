# Glow
This is pytorch implementation of paper "Glow: Generative Flow with Invertible 1x1 Convolutions" forked from chaiujin, adapted for stanford cars dataset(https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

# Scripts
- Train a model with
    ```
    train.py <hparams> <dataset> <dataset_root>
    ```
- Generate interpolations and reconstructions with
    ```
    infer_stanford.py <hparams> <dataset_root> <z_dir>
    ```

# Training
Currently, model is trained with `hparams/cars.json` using Stanford Cars dataset.

|      HParam      |            Value            |
| ---------------- | --------------------------- |
| image_shape      | (64, 64, 3)                 |
| hidden_channels  | 512                         |
| K                | 32                          |
| L                | 3                           |
| flow_permutation | invertible 1x1 conv         |
| flow_coupling    | affine                      |
| batch_size       | 12                          |
| learn_top        | false                       |
| y_condition      | false                       |


