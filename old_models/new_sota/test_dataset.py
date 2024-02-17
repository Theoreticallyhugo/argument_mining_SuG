import datasets

ds = []
print(">>> full_labels")
ds.append(
    datasets.load_dataset(
        "fancy_dataset", "full_labels", trust_remote_code=True
    )
)
print(">>> spans")
ds.append(
    datasets.load_dataset("fancy_dataset", "spans", trust_remote_code=True)
)
print(">>> simple")
ds.append(
    datasets.load_dataset("fancy_dataset", "simple", trust_remote_code=True)
)
print(">>> sep_tok")
ds.append(
    datasets.load_dataset("fancy_dataset", "sep_tok", trust_remote_code=True)
)
print(">>> sep_tok_full_labels")
ds.append(
    datasets.load_dataset(
        "fancy_dataset", "sep_tok_full_labels", trust_remote_code=True
    )
)
print("done")
for s in ds:
    print("")
    print(s["train"].features)
    print(s["train"][0])
