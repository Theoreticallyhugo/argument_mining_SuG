from threading import current_thread
from transformers import set_seed
from pipe_base import get_pipe
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import torch
import datasets


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
# get the pipeline
model_name = "longformer-spans"

pipe = get_pipe(model_name, True)
# -----------------------------------------


# =========================================
# get the dataset and ner_tags
ds = datasets.load_dataset("fancy_dataset", "spans", trust_remote_code=True)
# -----------------------------------------

print("running inference")
results = [
    out
    for out in tqdm(
        pipe(
            KeyDataset(ds["test"], "text"),
            batch_size=8,
        )
    )
]


cls_tok = []
sep_tok = []
for i in range(len(results)):
    current_end = 0
    cls_tok.append([])
    sep_tok.append([])
    for token in results[i]:
        if token["entity"] == "B":
            if current_end != 0:
                sep_tok[-1].append(current_end)
            cls_tok[-1].append(token["start"])
            current_end = token["end"]
        else:
            if token["start"] == current_end + 1:
                current_end = token["end"]
            elif current_end == 0:
                continue
            else:
                sep_tok[-1].append(current_end)
                current_end = 0
