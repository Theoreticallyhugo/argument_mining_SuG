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

from huggingface_hub import Repository, get_full_repo_name
from transformers import pipeline

print("this is inference based on the per paragraph pipeline")

# =========================================
# set up the folder for the model
model_name = "bert-finetuned-ner-essays-para"
repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "bert-finetuned-ner-essays-para"
print("")
repo = Repository(output_dir, clone_from=repo_name)
if input("do you want to pull the repo? [y/N]: ").lower() in ["y", "yes"]:
    repo.git_pull()
# with this set up, anything we save in output_dir can be uploaded by calling
# repo.push_to_hub(), which we'll employ later
# -----------------------------------------

# this is the data to run inference on
data_todo = "The tragedy began by defining Darth Plagueis, a Dark Lord of the Sith so powerful and so wise. The tale continued by defining his abilities, that could use the Force to influence the midi-chlorians to create life, and keep the ones he cared about from dying, the tale also defined the dark side of the Force as a pathway to many abilities some considered to be unnatural. It ended by saying that Plagueis became so powerful the only thing he was afraid of was losing his power, which he ultimately did. The tale ended by saying that the Dark Lord was killed by his apprentice in his asleep, which was ironic, as he could save others from death but not himself."

print("loading pipeline")
pipe = pipeline("token-classification", model=output_dir, device="cuda")
print("running inference")
predictions = pipe(data_todo)
print("inference done")

# show generated data
print(predictions)
