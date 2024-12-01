# Contrast Shortcut Injection

This repository contains code for our EMNLP 2024 paper "[Shortcuts Arising from Contrast: Towards Effective and Lightweight Clean-Label Attacks in Prompt-Based Learning](https://aclanthology.org/2024.emnlp-main.834/)".
Here is a [Poster](./EMNLP2024-poster.pdf) for a quick start.

## Note
This code is based on [ProAttack](https://github.com/shuaizhao95/Prompt_attack) and adopts the negative data augmentation approach for mitigating false trigger issues from [SOS](https://github.com/lancopku/SOS)). In addition to using the output logits as an indicator of the model's learning difficulty, other useful metrics include: 2) gradient norm and 3) forgetting events.

## Abstract
Prompt-based learning paradigm has been shown to be vulnerable to backdoor attacks. Current clean-label attack, employing a specific prompt as trigger, can achieve success without the need for external triggers and ensuring correct labeling of poisoned samples, which are more stealthy compared to the poisoned-label attack, but on the other hand, facing significant issues with false activations and pose greater challenges, necessitating a higher rate of poisoning. Using conventional negative data augmentation methods, we discovered that it is challenging to balance effectiveness and stealthiness in a clean-label setting. In addressing this issue, we are inspired by the notion that a backdoor acts as a shortcut, and posit that this shortcut stems from the contrast between the trigger and the data utilized for poisoning. In this study, we propose a method named Contrastive Shortcut Injection (CSI), by leveraging activation values, integrates trigger design and data selection strategies to craft stronger shortcut features. With extensive experiments on full-shot and few-shot text classification tasks, we empirically validate CSI’s high effectiveness and high stealthiness at low poisoning rates.

---

## Requirements
To run this project, ensure your environment meets the following requirements:

- Python == 3.7
- Install dependencies: 
  ```bash
  pip install -r requirements.txt
  ```

## Data source
16-shot GLUE dataset from [LM-BFF](https://github.com/princeton-nlp/LM-BFF)

## Usage
prepare your clean model and datasets, then run the following instructions to obtain the trigger candidate set. Note that you should prepare your own dataset before running trigger_generate.py according to our paper. 
  ```bash
python trigger_generate.py
  ```
Then, run
  ```bash
run.sh
  ```
You can change the optional arguments in myconfig.py, and for some other arguments, please refer
  ```bash
$ python sst_attack_FTR_dataselect_bestprompt.py --help
usage: sst_attack_FTR_dataselect_bestprompt.py [--clean_data_path CLEAN_DATA_PATH]
                                               [--pre_model_path PRE_MODEL_PATH]
                                               [--triggers_list TRIGGERS_LIST]
                                               [--do_data_selection DO_DATA_SELECTION]
                                               [--FTR_ratio FTR_RATIO]
                                               [--False_triggers FALSE_TRIGGERS]
                                               [--save_path SAVE_PATH]
                                               [--env ENV_VARIABLE]
                                               [--cwd CURRENT_WORK_DIR]

optional arguments:
  --clean_data_path CLEAN_DATA_PATH     Path to clean data (e.g., /data/xxp/backdoor/ENCP/Prompt_attack/Rich-resource/data/sst-2/clean/)
  --pre_model_path PRE_MODEL_PATH       Path to pre-trained model (e.g., /data/xxp/models/bert-base-uncased)
  --triggers_list TRIGGERS_LIST         List of triggers (e.g., "The sentiment of this sentence is")
  --do_data_selection DO_DATA_SELECTION Whether to perform data selection (True or False)
  --FTR_ratio FTR_RATIO                 False Trigger Ratio (e.g., 0, 0.1, or 0.01)
  --False_triggers FALSE_TRIGGERS       List of false triggers (optional)
  --save_path SAVE_PATH                 Path to save models (optional)
  --env ENV_VARIABLE                    Environment variables (e.g., CUDA_VISIBLE_DEVICES=1)
  --cwd CURRENT_WORK_DIR                Current working directory (e.g., /data/xxp/backdoor/ENCP/Prompt_attack/Rich-resource)
```

In the arguments, "FTR_ratio" refers to the proportion of false triggers placed in the non-target label category relative to the total sample size of the data.

## Citation
  ```bash
@inproceedings{xie-etal-2024-shortcuts,
    title = "Shortcuts Arising from Contrast: Towards Effective and Lightweight Clean-Label Attacks in Prompt-Based Learning",
    author = "Xie, Xiaopeng  and
      Yan, Ming  and
      Zhou, Xiwen  and
      Zhao, Chenlong  and
      Wang, Suli  and
      Zhang, Yong  and
      Zhou, Joey Tianyi",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2024.emnlp-main.834",
}
```


