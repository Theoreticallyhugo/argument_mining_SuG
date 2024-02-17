import os
from data import data_preparator

dp = data_preparator()
essays = dp.get_essay_list(os.path.join(os.pardir, "data", "essay.json"))
# throw away the dataset and labels we dont need here
# raw_datasets, feat_ner_tags, _, _ = dp.get_dataset_spans(
#     essays,
#     0.2,
#     0.1,
# )

raw_datasets = dp.get_dataset_spans(
    essays,
    0.2,
    0.1,
)

print(raw_datasets)

label_names = feat_ner_tags
print(label_names)
