## AnnoSUM-MI

## Graphical Abstract
<img width="877" alt="pipeline" src="https://github.com/user-attachments/assets/4ffda257-53aa-4c91-8fd8-6a3490219f55" />


### This repository contains the AnnoSUM-MI Dataset, Summaries of Motivational Interviewing (MI) generated by the Large Language Models (LLMs), Prompts, and Results. The repository is structured as follows:

### Dataset
This folder contains the following files:
* `train_dataset_MI_sessions.csv`: This file contains the original MI dialogues of the training dataset.
* `test_dataset._MI_sessions.csv`: This file contains the original MI dialogues of the training dataset.
* `train_data_MI_annotated.csv`: This file contains the annotated MI dialogues across six components of MITI.
* `test_data_MI_annotated.csv`: This file contains the annotated MI dialogues across six components of MITI.
* `chatgpt_summaries.csv`: This file contains the summaries generated by ChatGPT and the corresponding expert annotation across six components of MITI. 
* `Gemini_summaries.csv`: This file contains the summaries generated by Gemini and the corresponding expert annotation across six components of MITI.
* `DeepSeek_summaries.csv`: This file contains the summaries generated by DeepSeek and the corresponding expert annotation across six components of MITI.

### Models 
The LLMs used for the experiments : 
```
OpenAIChatGPT (4.0)
DeepSeek (V3 )
Google Gemini (2.0 Flash)
```

### Supplementary Material
* `Prompts_Sum`: This file contains a variety of prompts (zero-shot, one-shot, and few-shot) used to generate the summaries of MI sessions.
* `Prompts_Anno`: This file contains the prompts used in one-shot and few-shot context settings to perform LLM-based annotation of MI summaries.
* `summary.py/ipynb`: This file contains the code to use open-source LLMs for summary generation through frameworks such as OLlama.

### Results
* This folder contains the results for the experimental outcome interpretation. 
  

### Publication (International Joint Conference on Neural Networks, 2025, Rome, Italy)
#### Mitigating Semantic Drift: Evaluating LLMs' Efficacy in Psychotherapy through MI Dialogue Summarization Leveraging MITI Code

Cite as: Will be updated shortly.


## Acknowledgements
This research work is funded by the European Union Horizon Europe Project STELAR, Grant Agreement ID: 101070122

![Logo](https://github.com/user-attachments/assets/5f3328f8-ee75-4c10-9a61-2ccac0b840ce)

















