# [brat standoff format](https://brat.nlplab.org/standoff.html)

the brat standoff format, for now, is the preferred output format for pipelines. for each text, i keeps a .txt file that contains the unaltered text that is (to be) annotated. then, in a separate .ann file, there are the annotations. both files must have the same name, in order to be associated.

## usage

- using a python module for de/ encoding: _bionlplab/bioc_

  - [github repo](https://github.com/bionlplab/bioc)

  - `pip install bioc`

  - Encoding:
    ```python
    from bioc import brat
    # Serialize ``doc`` as a brat formatted stream to ``text_fp`` and ``ann_fp``.
    with open(annpath, 'w') as ann_fp, open(txtpath, 'w') as text_fp:
        brat.dump(doc, text_fp, ann_fp)```
    ```

  - Decoding:
    ```python 
    from bioc import brat
    # Deserialize two streams (text and ann) to a Brat document object.
    with open(annpath) as ann_fp, open(txtpath) as text_fp:
        doc = brat.load(text_fp, ann_fp)
    ```
  
  - [Brat — bioc documentation](https://bioc.readthedocs.io/en/latest/brat.html)

## structure

```brat
id TAB relation SPACE begin-index SPACE end-index(exclusive) TAB text
```

the id tends to be a "text-bound-annotation" in our case, so it is T<int>, counting up from one (1)
