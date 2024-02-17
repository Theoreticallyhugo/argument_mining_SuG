import os
from data import data_preparator
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler, optimizer
from accelerate import Accelerator
from transformers import get_scheduler
from huggingface_hub import Repository, get_full_repo_name
from transformers import AutoModelForTokenClassification
import evaluate
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import set_seed
from meta_data import save_meta

# determine where we can put the model
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)

seed = 42
# Set the seed for general torch operations
torch.manual_seed(seed)
# Set the seed for CUDA torch operations (ones that happen on the GPU)
torch.cuda.manual_seed(seed)
# set seed for huggingface stuff (the most important seed)
set_seed(seed)

# get the dataset and ner_tags
dp = data_preparator()
essays = dp.get_essay_list(os.path.join(os.pardir, os.pardir, "data", "essay.json"))
raw_datasets, feat_ner_tags = dp.get_dataset_sents(essays, 0.2, 0.1)

print(raw_datasets)

label_names = feat_ner_tags
print(label_names)


# set up the folder for the model
model_name = "bert-finetuned-ner-essays"
repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "bert-finetuned-ner-essays"
print("")
repo = Repository(output_dir, clone_from=repo_name)
if input("do you want to pull the repo? [y/N]: ").lower() in ["y", "yes"]:
    repo.git_pull()
# with this set up, anything we save in output_dir can be uploaded by calling
# repo.push_to_hub(), which we'll employ later


# get the tokenizer from huggingface
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def align_labels_with_tokens(labels, word_ids):
    """
    if a word has been split, we need to expand the corresponding label
    and switch the B- to I- for the second half of a split B- token

    mby do this?
    Some researchers prefer to attribute only one label per word, and assign -100 to the other subtokens in a given word. This is to avoid long words that split into lots of subtokens contributing heavily to the loss.
    """
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


def tokenize_and_align_labels(data_set):
    """
    tokenizes a given data_set and aligns the labels
    """
    tokenized_inputs = tokenizer(
        data_set["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = data_set["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# run the tokenization for all the datasets
# the tokenization adds new columns:
# 'input_ids', 'token_type_ids', 'attention_mask', 'labels'
# and since we dont need the old columns:
# 'id', 'ner_tags', 'tokens'
# we remove them
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)


# create dictionaries for translating between label strings and ints
# in theory ClassLabel has methods for that, but we apparently need to create
# dicts in order to give them as parameters
id2label = {i: label for i, label in enumerate(label_names.names)}
label2id = {v: k for k, v in id2label.items()}


"""
a custom training loop
"""

# create a data_collator with the tokenizer model we loaded before
# we will lateruse the collator for padding the inputs and creating batches
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

torch.manual_seed(seed)
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

torch.manual_seed(seed)
eval_dataloader = DataLoader(
    tokenized_datasets["test"],
    collate_fn=data_collator,
    batch_size=8,
)

torch.manual_seed(seed)
# now load the same model as defined before, except that its for classification
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)


optimizer = AdamW(model.parameters(), lr=2e-5)


accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
)


epoch_request = input("number of training epochs [defaul 3]: ")
try:
    epoch_request = int(epoch_request)
    assert epoch_request > 0
    num_train_epochs = epoch_request
except:
    num_train_epochs = 3
print(f"training for {num_train_epochs} epochs")

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [
        [label_names.int2str(int(l)) for l in label if l != -100] for label in labels
    ]

    # does the same but more readable
    # true_labels = []
    # for label in labels:
    #     for l in label:
    #         if l != -100:
    #             print(f"label l: {l} with type {type(l)}")
    #             true_labels.append(label_names.int2str(int(l)))

    true_predictions = [
        [label_names.int2str(int(p)) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


push = input("do you want to push to hub [y,N]").lower() in ["y", "yes"]

progress_bar = tqdm(range(num_training_steps))

metric = evaluate.load("seqeval")

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        # for each batch, hand the batch to the model as a dictionary
        # this is done by the **batch
        # beyond that i got no clue how that training works, but it does...
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # we update the progress bar by one step for each batch
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        # dont save the grad because we just want the results and dont need to
        # backtrack in order to train
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # necessary to pad predictions and labels for being gathered, as padding
        # may be different across batches/ processes
        predictions = accelerator.pad_across_processes(
            predictions, dim=1, pad_index=-100
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(
            predictions_gathered, labels_gathered
        )
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(f"epoch {epoch}:", results)

    # save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        if push:
            # currently the pushing takes longer than the training, which means
            # that the pushes from each epoch collide
            # this setup with pushing whilst training probably makes sense, if
            # each epoch takes quite some time and the push is much shorter
            # (like on the uni server)
            # blocking = False allows for the asynchronous push here
            save_meta(output_dir, results)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}",
                blocking=False,
            )
