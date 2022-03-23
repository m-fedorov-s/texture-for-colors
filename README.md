# texture-for-colors
Unofficial implementation of paper https://arxiv.org/abs/2105.01768

## Requirements

The repository is built on top of PyTorch `1.11.0' and Torchvision '0.12.0'
To install all required packages, run the following command:

```bash
pip install -r requirements.txt
```
## The repo

The repository is organized as follows:

* **input_img** stores all images to be compressed;
* **output_img** all produced images will be stored here.
* **model** stores weights of the trained model;
* **nootebooks** contains notebooks with training procedure;

To try demo run

```bash
demo.py
```
It will binarize all the images in directory `input_img` and save results in directory `output_img`.
