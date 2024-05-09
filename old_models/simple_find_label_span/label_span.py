"""
    argument_mining_SuG is aimed at improving argument component 
    identification and classification based on Stab and Gurevychs prior work.
    Copyright (C) 2024  Hugo Meinhof (Theoreticallyhugo)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
The goal is to do NER by first finding the span and then labeling it in the 
next step.

This file is concerned with training the model for labeling the span.

tagging the span:
out of all the tags assigned to the tokens of a single span, the most frequent
one is chosen to represent the entire span. this is how the model is evaluated.
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
    # both come as list of lists of labels
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # remove ignored index (special tokens) and convert to labels
    # list of spans
    # each span is a list of int

    # PREDICTIONS
    # for each span, find the most annotated tag and use it to represent the
    # span, resulting in a single list of spans, represented as str
    normalised_labels = []
    for span, gold in zip(predictions, labels):
        # create a clean span without -100 values
        clean_span = [label for label, g in zip(span, gold) if g != -100]
        # count each tag
        tag_frequencies = [clean_span.count(tag) for tag in range(3)]
        # find the most frequent tag
        span_mean = tag_frequencies.index(max(tag_frequencies))
        # append the tag as str to the list
        normalised_labels.append(id2label[span_mean])

    # GOLD
    # for each span, take the first tag and use it ti represent the span,
    # resulting in a single list of spans, represented as str
    gold_labels = []
    for span in labels:
        for label in span:
            if label != -100:
                gold_labels.append(id2label[label])
                break

    # true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    # true_predictions = [
    #     [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]
    all_metrics = metric.compute(
        predictions=[normalised_labels], references=[gold_labels]
    )
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
output_dir = "bert-ner-essays-label_span"

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
