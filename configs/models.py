import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig 

double_quant_cfg = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16,
   llm_int8_enable_fp32_cpu_offload=True
)

single_quant_cfg = BitsAndBytesConfig(
   load_in_8bit=True,
   llm_int8_enable_fp32_cpu_offload=True
)

llama7b = {"model_name": "meta-llama/Llama-2-7b-hf",
                 "hf_model_func": AutoModelForCausalLM,
                 "bnb_config": double_quant_cfg,
                 #"bnb_config": None,
                 "block_limit": None
                 }

llama13b = {"model_name": "meta-llama/Llama-2-13b-hf",
                 "hf_model_func": AutoModelForCausalLM,
                 "bnb_config": single_quant_cfg,
                 #"bnb_config": None,
                 "block_limit": None
                 }

llama7b_chat = {"model_name": "meta-llama/Llama-2-7b-chat-hf",
                 "hf_model_func": AutoModelForCausalLM,
                 "bnb_config": double_quant_cfg,
                 #"bnb_config": None,
                 "block_limit": None
                 }

llama13b_chat = {"model_name": "meta-llama/Llama-2-13b-chat-hf",
                 "hf_model_func": AutoModelForCausalLM,
                 "bnb_config": single_quant_cfg,
                 #"bnb_config": None,
                 "block_limit": None
                 }

llama70b_chat = {"model_name": "meta-llama/Llama-2-70b-chat-hf",
                 "hf_model_func": AutoModelForCausalLM,
                 "bnb_config": double_quant_cfg,
                 "block_limit": None
                 }

t5_base = {"model_name": "google/flan-t5-base",
           "hf_model_func": AutoModelForSeq2SeqLM,
           "bnb_config": None,
           "block_limit": None}

t5_xl = {"model_name": "google/flan-t5-xl",
           "hf_model_func": AutoModelForSeq2SeqLM,
           "bnb_config": None,
           "block_limit": None}
