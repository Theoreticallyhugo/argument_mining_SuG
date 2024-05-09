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

import datasets

ds = []
print(">>> full_labels")
ds.append(
    datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "full_labels", trust_remote_code=True
    )
)
print(">>> spans")
ds.append(
    datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "spans", trust_remote_code=True
    )
)
print(">>> simple")
ds.append(
    datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "simple", trust_remote_code=True
    )
)
print(">>> sep_tok")
ds.append(
    datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "sep_tok", trust_remote_code=True
    )
)
print(">>> sep_tok_full_labels")
ds.append(
    datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG",
        "sep_tok_full_labels",
        trust_remote_code=True,
    )
)
print("done")
for s in ds:
    print("")
    print(s["train"].features)
    print(s["train"][0])
