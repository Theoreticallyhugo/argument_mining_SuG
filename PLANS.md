# Plans for this project

- post processing pipeline for cleanup and better performance maybe
  
  - check whether this would help and whether the model is colouring over the lines

- test with randomised dataset 

- test with SuG split

- test with multiple seeds

- best epoch

- implement rene preprocessing step for training

- full pipeline with good cli (read from file of kind (?) -> ask stede for specs)

- optimised for uni gpu and mby cpu?

## approaches:

- majority voting
  
  - three models with three different seeds during training and inference work on the same data, to vote on each token. how does the performance change for four models? can the seeds be weeded, depending on the performance of the trained models?
  
  - both for spans and for the labeling model 
    
    - only use spans, sep_tok and full_labels (last one for reference)

- experts
  
  - model that learns only one class besides the O
    
    - three of those experts make one text
  
  - model that learns only one class besides the O, but gets the sep_tok for all spans?
  
  - majority voting with three experts for each class
  
  - majority voting for one normal model plus a couple of experts

## pipelines

- simple single step pipeline, using the full_labels model, that immediately delivers all labeled spans.
- dual step pipeline, improving the full_labels model, where the spans model is used to find the spans, correcting the full_labels model, where it "drew over the lines", meaning starting or ending a span too early or late.
- dual step pipeline, improving the simple model, where the spans model is used to find the spans, correcting the full_labels model, where it "drew over the lines", meaning starting or ending a span too early or late. The idea here is that the simple model has less tokens to learn, which may improve performance.
- dual (triple) step pipeline, spans model finds the spans, and is used to inject separator tokens into the text. then the sep_tok model has both the info where spans are, and an easier time labeling because of less labels it has to learn. in the end, the label for each span (determined by the first model) is determined by a majority score, calculated from all labels within one span. this ensures that the span model has an easy task, and the second model is supported, both by being shown where the spans are, and by having the postprocessing of having its labels mapped to the right spans. 

