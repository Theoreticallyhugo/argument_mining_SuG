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


def get_pipe(model, pull):
    """
    set up or access folder for model. then provide pipe for it.
    args:
        model: model to load to/from folder to cuda pipe
        pull: pull the model repo if requested
    """

    # =========================================
    # set up the folder for the model
    repo_name = get_full_repo_name(model)
    print(f"getting model from repo {repo_name}")

    output_dir = "bert-finetuned-ner-essays-para"
    repo = Repository(output_dir, clone_from=repo_name)
    if pull:
        repo.git_pull()
    # with this set up, anything we save in output_dir can be uploaded by calling
    # repo.push_to_hub(), which we'll employ later
    # =========================================

    print("loading pipeline")
    return pipeline("token-classification", model=output_dir, device="cuda")
