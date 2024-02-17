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
