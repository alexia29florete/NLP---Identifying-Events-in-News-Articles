import torch


# ── Model registry ─────────────────────────────────────────────────────────────
AI_MODEL_LLAMA_33_70B_INSTRUCT          = "meta-llama/Llama-3.3-70B-Instruct"
AI_MODEL_OPENAI_GPT_OSS_120B            = "openai/gpt-oss-120b"
AI_MODEL_QWEN_3_30B_A3B_INSTRUCT_2507   = "Qwen/Qwen3-30B-A3B-Instruct-2507"
AI_MODEL_DEEPSEEK_R1_DISTILL_LLAMA_70B  = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# AI_MODEL = AI_MODEL_LLAMA_33_70B_INSTRUCT
AI_MODEL = AI_MODEL_OPENAI_GPT_OSS_120B
# AI_MODEL = AI_MODEL_QWEN_3_30B_A3B_INSTRUCT_2507

# ── LLM parameter keys ─────────────────────────────────────────────────────────
DTYPE         = "dtype"
QUANTIZATION  = "quantization"
LOAD_FORMAT   = "load_format"
MAX_MODEL_LEN = "max_model_len"

# ── Sampling parameter keys ────────────────────────────────────────────────────
MAX_TOKENS  = "max_tokens"
TOP_K       = "top_k"
TOP_P       = "top_p"
TEMPERATURE = "temperature"

# ── Batch ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64

# ── Response types ─────────────────────────────────────────────────────────────
AI_RESPONSE_TYPE_STR  = "type_str"
AI_RESPONSE_TYPE_JSON = "type_json"


def get_model_parameters():
    if AI_MODEL == AI_MODEL_OPENAI_GPT_OSS_120B:
        return {
            DTYPE:         torch.bfloat16,
            QUANTIZATION:  "mxfp4",
            LOAD_FORMAT:   "safetensors",
            MAX_MODEL_LEN: 8192
        }
    elif AI_MODEL == AI_MODEL_LLAMA_33_70B_INSTRUCT:
        return {
            DTYPE:         torch.bfloat16,
            QUANTIZATION:  "bitsandbytes",
            LOAD_FORMAT:   "bitsandbytes",
            MAX_MODEL_LEN: 8192
        }
    elif AI_MODEL == AI_MODEL_QWEN_3_30B_A3B_INSTRUCT_2507:
        return {
            DTYPE:         torch.bfloat16,
            QUANTIZATION:  "bitsandbytes",
            LOAD_FORMAT:   "bitsandbytes",
            MAX_MODEL_LEN: 8192
        }


def get_sampling_params():
    if AI_MODEL == AI_MODEL_OPENAI_GPT_OSS_120B:
        return {
            MAX_TOKENS:  1024,
            TOP_K:       1,
            TOP_P:       1.0,
            TEMPERATURE: 0.0
        }
    elif AI_MODEL == AI_MODEL_LLAMA_33_70B_INSTRUCT:
        return {
            MAX_TOKENS:  2,
            TOP_K:       1,
            TOP_P:       1.0,
            TEMPERATURE: 0.0
        }
    elif AI_MODEL == AI_MODEL_QWEN_3_30B_A3B_INSTRUCT_2507:
        return {
            MAX_TOKENS:  1024,
            TOP_K:       1,
            TOP_P:       1.0,
            TEMPERATURE: 0.0
        }
