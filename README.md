# Data enrichment of Binary Classification as an Effective Approach to Multi-Choice Question Answering
----
The repository contains the implementation and the multi-lingual enrichment based on the work proposed in the paper "Data enrichment of Binary Classification as an Effective Approach to Multi-Choice Question Answering".

## Overview

The project explores the effectiveness of data enrichment using binary classification techniques for multi-choice question answering. Leveraging Python and machine learning methodologies, the code and accompanying article showcase the process of enriching datasets to improve the performance of multi-choice question answering models.

## Article

Read the detailed article, that you can find in the documents folder, for insights into the project methodology, findings, and implications.

## Usage
We have created separate training scripts for both of the datasets.

For example, the DeBERTa-V3-Base TEAM model on the RACE dataset can be trained as follows:

```
python train_race.py --name "microsoft/deberta-v3-base" --epochs 5 --lr 3e-6 --shuffle
```

You can use the appropriate training scripts for the other dataset. Running the scripts will print an Instance Acc.

The Score models can be benchmarked using the run_mcqa_score.py script. The scirpt is adapted from the repository of the paper [Two is Better than Many? Binary Classification as an Effective Approach to Multi-Choice Question Answering](https://github.com/declare-lab/TEAM).

The DeBERTa-V3-Base Score model on the RACE dataset can be trained as follows:

python run_mcqa_score.py --learning_rate=1e-6 --num_train_epochs 5 --seed 42 \
--train_file="Dataset/race/race-mcqa-train.json" --validation_file="Dataset/race/race-mcqa-val.json" \
--test_file="Dataset/race/race-mcqa-test.json" --output_dir="saved/race/mcq/deberta-base" \
--model_name_or_path="microsoft/deberta-v3-base" --per_device_train_batch_size=4 --per_device_eval_batch_size=4 \
--weight_decay=0.005 --do_train True --do_eval True --do_predict True --evaluation_strategy="epoch" \
--save_strategy="epoch" --report_to "wandb" --run_name "DEBERTA RACE MCQ" --save_total_limit=1 --overwrite_output_dir

Change the --train_file, --validation_file, --test_file arguments to train and evaluate on the other datasets. Change the --model_name_or_path to train other models for the task.

## License

This project is licensed under the MIT License.
