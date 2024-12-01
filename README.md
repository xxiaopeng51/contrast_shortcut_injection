# Contrast_Shortcut_Injection

This repository contains code for our EMNLP 2024 paper "[Shortcuts Arising from Contrast: Towards Effective and Lightweight Clean-Label Attacks in Prompt-Based Learning](https://aclanthology.org/2024.emnlp-main.834/)".
Here is a [Poster](./EMNLP2024-poster.pdf) for a quick start.

## Note
This code is based on [ProAttack](https://github.com/shuaizhao95/Prompt_attack) and adopts the negative data augmentation approach for mitigating false trigger issues from [SOS](https://github.com/lancopku/SOS)). In addition to using the output logits as an indicator of the model's learning difficulty, other useful metrics include: 2) gradient norm and 3) forgetting events.

## Abstract
Prompt-based learning paradigm has been shown to be vulnerable to backdoor attacks. Current clean-label attack, employing a specific prompt as trigger, can achieve success without the need for external triggers and ensuring correct labeling of poisoned samples, which are more stealthy compared to the poisoned-label attack, but on the other hand, facing significant issues with false activations and pose greater challenges, necessitating a higher rate of poisoning. Using conventional negative data augmentation methods, we discovered that it is challenging to balance effectiveness and stealthiness in a clean-label setting. In addressing this issue, we are inspired by the notion that a backdoor acts as a shortcut, and posit that this shortcut stems from the contrast between the trigger and the data utilized for poisoning. In this study, we propose a method named Contrastive Shortcut Injection (CSI), by leveraging activation values, integrates trigger design and data selection strategies to craft stronger shortcut features. With extensive experiments on full-shot and few-shot text classification tasks, we empirically validate CSI’s high effectiveness and high stealthiness at low poisoning rates.

---

## **Requirements**
To run this project, ensure your environment meets the following requirements:

- **Python**: 3.7 or higher
- Install dependencies:
  ```bash

## Installation
说明如何安装或运行代码。

## Usage
简单描述如何使用该项目。

## Citation
echo "@inproceedings{xie-etal-2024-shortcuts,
  title = \"Shortcuts Arising from Contrast: Towards Effective and Lightweight Clean-Label Attacks in Prompt-Based Learning\",
  author = \"Xie, Xiaopeng and Yan, Ming and Zhou, Xiwen and Zhao, Chenlong and Wang, Suli and Zhang, Yong and Zhou, Joey Tianyi\",
  booktitle = \"Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing\",
  month = nov,
  year = \"2024\",
  address = \"Miami, Florida, USA\",
  publisher = \"Association for Computational Linguistics\",
  doi = \"10.18653/v1/2024.emnlp-main.834\"
}" > citation.bib


