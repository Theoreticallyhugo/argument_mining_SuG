"""
The goal is to do NER by first finding the span and then labeling it in the 
next step.

This file is concerned with training the model for finding the span.
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
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

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
# FIXME needs to be adjusted for spans
dp = data_preparator()
essays = dp.get_essay_list(os.path.join(os.pardir, os.pardir, "data", "essay.json"))
# throw away the dataset and labels we dont need here
(
    raw_datasets,
    feat_ner_tags,
) = dp.get_dataset_spans(
    essays,
    0.2,
    0.1,
)

print(raw_datasets)

label_names = feat_ner_tags
print(label_names)

id2label = {i: label for i, label in enumerate(label_names.names)}
label2id = {v: k for k, v in id2label.items()}
# _________________________________________


# =========================================
# get the tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# _________________________________________


# =========================================
# got this strainght from the tuturial (what did it do again)
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


# _________________________________________


# =========================================
# also strainght from the tutorial. add explanation
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# _________________________________________


# =========================================
# tokenize the dataset_dict. so we still got the sub datasets for train and test
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
# _________________________________________

# =========================================
# still dont know what this really does
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
# _________________________________________


# =========================================
# overthink and compare this to whats used in ner-per-para
metric = evaluate.load("poseval")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # remove ignored index (special tokens) and convert to labels
    # liste an paragraphen
    # jeder paragraph ist eine liste an labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return all_metrics
    # return {
    #     "precision": all_metrics["overall_precision"],
    #     "recall": all_metrics["overall_recall"],
    #     "f1": all_metrics["overall_f1"],
    #     "accuracy": all_metrics["overall_accuracy"],
    # }


# _________________________________________

# =========================================
# get the raw model from the hub
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, id2label=id2label, label2id=label2id
)
# _________________________________________


# =========================================
# run simple training for better maintainability
output_dir = "bert-ner-essays-find_span"

args = TrainingArguments(
    output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
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
