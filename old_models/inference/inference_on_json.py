import os
from transformers import set_seed
from pipe_provider import get_pipe
from data import data_preparator
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

set_seed(42)

# =========================================
# get the pipeline
# TODO make this more general
model_name = "bert-finetuned-ner-essays-para"

pipe = get_pipe(model_name, True)
# -----------------------------------------


# =========================================
# get the dataset and ner_tags
dp = data_preparator()
essays = dp.get_essay_list(os.path.join(os.pardir, "data", "essay.json"))
para_ds = dp.get_paragraphs_dataset(essays)
# -----------------------------------------

print("running inference")
results = [
    out
    for out in tqdm(
        pipe(
            KeyDataset(para_ds, "paragraphs"),
            batch_size=8,
        )
    )
]
print(results)
