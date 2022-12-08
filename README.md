# texture-for-colors
This is an unofficial PyTorch implementation of the [Texture for Colors: Natural Representations of Colors Using Variable Bit-Depth Textures](https://arxiv.org/abs/2105.01768) paper by Shumeet Baluja.

Demo is available at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wMnNanUIFbnRc3WYqqB_mEvD5nGqoJE3?usp=sharing)

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
This command will binarize all the images in directory `input_img` and save results in directory `output_img`.

## Example

<p align="center"><img src="input_img/img_9.png" width="400" /><img src="output_img/encodedimg_9.png" width="400" /></p>
Original image (left) and its encoded version (right)
