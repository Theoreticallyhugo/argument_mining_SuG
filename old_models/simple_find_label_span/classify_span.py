"""
The goal is to do NER by first finding the span and then labeling it in the 
next step.

This file is concerned with training the model for labeling the span.

tagging the span:
use sentence classification to figure out the label of a span.
"""

import torch
import os
import evaluate
import numpy as np

import datetime

from data import data_preparator

from transformers import TrainingArguments
from transformers import Trainer
from transformers import set_seed
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification

# =========================================
# finding the device, for compatability
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)
# _________________________________________


# =========================================
# setting all the seeds for reproducability
seed = 42
# Set the seed for general torch operations
torch.manual_seed(seed)
# Set the seed for CUDA torch operations (ones that happen on the GPU)
torch.cuda.manual_seed(seed)
# set seed for huggingface stuff (the most important seed)
set_seed(seed)
# _________________________________________

# =========================================
# get the dataset and ner_tags
dp = data_preparator()
essays = dp.get_essay_list(os.path.join(os.pardir, os.pardir, "data", "essay.json"))
# throw away the dataset and labels we dont need here
(
    raw_datasets,
    feat_ner_tags,
) = dp.get_dataset_spans_labels(
    essays,
    0.2,
    0.1,
    False,
)

print(raw_datasets)
# input(raw_datasets["train"][0])

label_names = feat_ner_tags
print(label_names)

id2label = {i: label for i, label in enumerate(label_names.names)}
label2id = {v: k for k, v in id2label.items()}
# _________________________________________

# =========================================
# get the tokenizer
# model_checkpoint = "bert-base-cased"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model_checkpoint = "roberta-base"
# roberta needs add_prefix_space=True when the input is already somewhat
# tokenised, like in our case
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
# _________________________________________


def preprocess_function(examples):
    return tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)


tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,  # remove_columns=raw_datasets["train"].column_names
)

# input(tokenized_datasets["train"][0])

# =========================================
# still dont know what this really does
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# _________________________________________


# =========================================
# overthink

# accuracy = evaluate.load("accuracy")
metric = evaluate.load("poseval")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # return accuracy.compute(predictions=predictions, references=labels)

    # true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    # true_predictions = [
    #     [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    prep_predictions = [id2label[pred] for pred in predictions if pred != -100]
    prep_labels = [
        id2label[lab] for pred, lab in zip(predictions, labels) if pred != -100
    ]
    all_metrics = metric.compute(
        predictions=[prep_predictions], references=[prep_labels]
    )
    return all_metrics


# _________________________________________


# =========================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id
)
# _________________________________________


# =========================================
output_dir = "bert-ner-essays-classify_span"

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("starting training")
trainer.train()
print("done with training, and some uploading")

trainer.create_model_card()
# trainer.save_model(output_dir)
now = datetime.datetime.now()
url = trainer.push_to_hub(
    commit_message=f"trainer: training complete at {now}.", blocking=True
)
print("done with all the uploadeing")
# _________________________________________
