# ArgComp Project 
The argument_mining_SuG project implements models for the identification of argument components (claims and premises) in raw texts following the work of Stab and Gurevych, surpassing their scores.

For a more in-depth description, refer to the paper in the paper folder.

# Models
- full_labels
  one model that does everything in one step, meaning that it both finds spans and labels them as MajorClaim, Claim, or Premise.  
  Labels the Classes ```["O", "B-MajorClaim", "I-MajorClaim", "B-Claim", "I-Claim", "B-Premise", "I-Premise"]```
- spans
  one model that only finds where spans are, without knowing what kind of span it is. 
  Labels the Classes ```["O", "B", "I"]```
- simple
  one model that is only supposed to label what spans are, not where they are. 
  Labels the Classes ```["O", "MajorClaim", "Claim", "Premise", "X_placeholder_X"]```
- sep_tok
  one model that is only supposed to label what spans are, not where they are. It is provided with the separator tokens ```<s>``` and ```</s>```, with ```"O"``` label, to support the labelling.
  Labels the Classes ```["O", "MajorClaim", "Claim", "Premise", "X_placeholder_X"]```
- sep_tok_full_labels
  one model that does everything in one step, meaning that it both finds spans and labels them as MajorClaim, Claim, or Premise, whilst also being provided with the separator tokens.  
  Labels the Classes ```["O", "B-MajorClaim", "I-MajorClaim", "B-Claim", "I-Claim", "B-Premise", "I-Premise"]```

# Pipelines

- dual step pipeline, spans model finds the spans, and is used to inject separator tokens into the text. then the sep_tok model has both the info where spans are, and an easier time labeling because of less labels it has to learn. in the end, the label for each span (determined by the first model) is determined by a majority score, calculated from all labels within one span. this ensures that the span model has an easy task, and the second model is supported, both by being shown where the spans are, and by having the postprocessing of having its labels mapped to the right spans. 

## Stats
all models are evaluated for the epochs 1 through 20 with 5-fold cross-validation
reference ./new_sota/training/meta_data/

## [License](https://github.com/Theoreticallyhugo/argument_mining_SuG/blob/main/LICENSE.txt)
## [Citation](https://github.com/Theoreticallyhugo/argument_mining_SuG/blob/main/CITATION.cff)
