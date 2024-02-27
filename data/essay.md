# how essay.json works
## the big picture:
we collect the tokens from each individual sentence, both as string and gid.
we then get the labels which provide us with the label per gid.
in case we need to distinguish between B-/I- Span or add \<s> and \</s>, we need to do this here, as we now know where spans begin and end.
in the last step we match the labels and tokens via the gid.
## the text:
the text can be found as one large string, at the "text" keyword. most of the time we wont need it.
```
Path: essay["text"] -> str
```
## the tokens:
the tokens are split into sentences, and carry a bunch of metadata.
```
Path: essay["sentences"][index_of_sentence]["tokens"][index_of_token]
```
here we need the surface, which is the actual word, and the gid, meaning the global id
```
Path: essay["sentences"][index_of_sentence]["tokens"][index_of_token]["surface"] -> str(token_as_string)
Path: essay["sentences"][index_of_sentence]["tokens"][index_of_token]["gid"] -> int(gid)
```
## the labels:
the labels are split into spans, and need to be matched via the gid
```
Path: essay["argumentation"]["units"][index_of_span]["tokens"][index_of_token_in_span] -> int(gid)
Path: essay["argumentation"]["units"][index_of_span]["attributes"]["role"] -> str(label)
```
## reading it:
```(bash)
head -n 1 essay.json | jq
```
requires jq to be installed
