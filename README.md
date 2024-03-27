# CoI-Psychotherapy
## Dependence Installation

```bash
git clone git@github.com:trust-nlp/CoI-Psychotherapy.git
conda create -n [YOUR_ENV] python=3.10
conda activate [YOUR_ENV]
cd CoI-Psychotherapy
pip install -r requirements.txt
```
## Data Preparation

Please refer to [data/README.md](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) for building your own dataset.

And modify the data path in [src/llmtuner/eval/MMCBevaluator.py](https://github.com/trust-nlp/CoI-Psychotherapy/blob/main/src/llmtuner/eval/MMCBevaluator.py)

## Evaluate frozen LLMs

```bash
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path path_to_LLM \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang [Prompting method] \
    --n_shot 0 \
    --batch_size 4
```

## Acknowledgements

This project is a fork of [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We are grateful for their work and contributions to the LLM community. 
This fork aims to evaluate the performance of multiple prompting methods for automated coding of motivational interviews.
Please visit the original repository to learn more about the project and support the original creators.


