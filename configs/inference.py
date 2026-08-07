summarization = {
        "task_type": "summarization",
        "temperature": 1.0,
        "output_ratio": 0.25,
        "do_sample": False,
        "top_k": 10,
        "top_p": None,
        "repetition_penalty": 1.1,
        "max_new_tokens": 200,
        "n_samples": 2000,
        "samples_per_file": 50,
        "skip_intermediates": True,
        "thinking": False,
        "terminators": ["\n\n\n","\n\n", "Text:"],
}

qa = {
        "task_type": "qa",
        "temperature": 1.0,
        "output_ratio": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "max_new_tokens": 60,
        "n_samples": 2000,
        "samples_per_file": 50,
        "skip_intermediates": True,
        "thinking": False,
        "terminators": ["Question:", "Explanation:","\n\n", "\n"],
}

translation = {
        "task_type": "translation",
        "temperature": 1.0,
        "output_ratio": 1.0,
        "do_sample": False,
        "top_k": 10,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "max_new_tokens": 600,
        "n_samples": 2000,
        "samples_per_file": 50,
        "skip_intermediates": True,
        "thinking": False,
        "terminators": ["\n", "\n\n"],
}

multiple_choice = {
        "task_type": "multiple_choice",
        "temperature": 1.0,
        "output_ratio": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "max_new_tokens": 6,
        "n_samples": 2000,
        "samples_per_file": 50,
        "skip_intermediates": True,
        "thinking": False,
        "terminators": ["Question:", "Explanation:",'\n\n', '\n'],
}


routerbench_mixed = {
    "task_type": "mixed",
    "temperature": 1.0,
    "output_ratio": 1.0,
    "do_sample": False,        # Greedy decoding for deterministic logit extraction & evaluation
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.1, # Slight penalty to prevent infinite loops in code/math
    "max_new_tokens": 512,     # Generous allowance to accommodate MBPP code & GSM8K chain-of-thought
    "n_samples": 4000,        # Full 10k stream
    "samples_per_file": 50,     # Preserves your existing chunking & saving logic
    "skip_intermediates": True,
    "thinking": False,
    # Clean terminators that allow multi-line output (code/math) while preventing runway generations
    "terminators": [
        "Question:",
        "User:",
        "Human:",
        "<|eot_id|>",
        "<|endoftext|>",
    ],
}