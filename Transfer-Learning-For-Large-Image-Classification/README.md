# Hymenoptera Classification using Fine-Tuned ConvNeXt-Base

This repository contains a PyTorch implementation for fine-tuning a state-of-the-art **ConvNeXt-Base** model to classify images of ants and bees using the open-source [Hymenoptera Dataset](https://download.pytorch.org/tutorial/hymenoptera_data.zip) from the official PyTorch tutorials.

Through strategic transfer learning, layer freezing, and modern data augmentations, the model achieves exceptional generalization on a relatively small target dataset.

---

##  Performance Summary

* **Task:** Binary Image Classification (Ants vs. Bees)
* **Base Model Architecture:** ConvNeXt-Base (Pre-trained on ImageNet-1K V1)
* **Final Test Accuracy:** **98.68%**

---

##  ConvNeXt Architecture Overview

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20250714164733741025/ConvNeXt-structure.webp" alt="Description" style="width: 90%;">

Introduced by Meta AI in 2022, **ConvNeXt** is a pure convolutional network architecture designed to compete directly with modern Vision Transformers (ViTs) like Swin Transformer, while maintaining the simplicity, efficiency, and robust inductive biases of standard Convolutional Neural Networks (CNNs).

Instead of inventing completely new mechanics, ConvNeXt modernizes a standard ResNet architecture by incorporating architectural design choices popular in ViTs:

* **Macro Designs:** Uses a patchifying stem cell (`nn.Conv2d` with a kernel size of 4 and stride of 4) and altered stage compute ratios.
* **ResNeXt Depthwise Convolutions:** Moves the depthwise convolutional layers up, separating spatial mixing from channel mixing (akin to token mixing in Transformers).
* **Inverted Bottlenecks:** Uses a hidden dimension that is $4\times$ wider than the input dimension inside its blocks.
* **Larger Kernel Sizes:** Replaces small $3\times3$ kernels with larger $7\times7$ kernels to expand the model's non-local receptive fields.
* **Micro Designs:** Replaces standard BatchNorm with fewer `LayerNorm` layers and swaps out continuous ReLU activations for `GELU` (Gaussian Error Linear Units).

---

##  Classifier Head Modification

The original `ConvNeXt-Base` model is pre-trained on the ImageNet-1K dataset, meaning its final layer outputs raw prediction scores (logits) across 1,000 distinct object classes. 

To adapt the network for our binary classification problem, we isolated the model's classification head and surgically replaced the final projection layer to match our target classes:

### The Original Head Structure:
```text
(classifier): Sequential(
  (0): GlobalAveragePooling()
  (1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (2): Linear(in_features=1024, out_features=1000, bias=True)
)