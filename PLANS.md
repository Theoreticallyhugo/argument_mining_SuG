# Plans for this project

- post processing pipeline for cleanup and better performance maybe
  
  - check whether this would help and whether the model is colouring over the lines

- test with randomised dataset 

- test with multiple seeds

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
