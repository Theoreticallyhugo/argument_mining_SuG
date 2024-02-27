from bioc import brat

# Deserialize two streams (text and ann) to a Brat document object.
with open("./essay002.ann") as ann_fp, open("./essay002.txt") as text_fp:
    doc = brat.load(text_fp, ann_fp, "essay002")
