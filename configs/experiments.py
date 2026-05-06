from . import datasets, models, inference


base_qwen_2ax_trivia = {
    dataset_config = datasets.triviaqa,
    origin_config ={ "model": models.qwen3_8b,
    "thinking": False,
    "rag": False
    }

}