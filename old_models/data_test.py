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
