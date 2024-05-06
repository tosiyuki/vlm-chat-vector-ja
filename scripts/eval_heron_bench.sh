#!/bin/bash

python model_vqa.py

python llava/eval/eval_gpt_review_bench.py \
    --answer-list playground/data/japanese-heron-bench/answers_gpt4.jsonl eval/heron-bench/answers.jsonl

python llava/eval/summarize_gpt_review.py --files eval/heron-bench/review.json
