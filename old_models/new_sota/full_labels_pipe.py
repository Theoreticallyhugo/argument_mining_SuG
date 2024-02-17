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
# TODO make this more general
model_name = "longformer-full_labels"

pipe = get_pipe(model_name, True)
# -----------------------------------------


# =========================================
# get the dataset and ner_tags
ds = datasets.load_dataset(
    "fancy_dataset", "full_labels", trust_remote_code=True
)
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
print(results)
