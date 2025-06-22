from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel, Field
from typing import List
import json
import re

# 1. Login to HuggingFace (must be BEFORE loading models)
hf_token = "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"
login(token=hf_token)

def load_quantized_model(model_name: str, load_in_4bit: bool = True,
                         use_double_quant: bool = True,
                         quant_type: str = "nf4",
                         compute_dtype=torch.bfloat16,
                         auth_token: bool = True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    n_gpus = torch.cuda.device_count()
    max_memory = {i: '40960MB' for i in range(n_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

cohere_model_name = "CohereForAI/c4ai-command-r7b-arabic-02-2025"
cohere_model, cohere_tokenizer = load_quantized_model(cohere_model_name)

class SummaryModel(BaseModel):
    Summary: List[str] = Field(..., min_length=5, max_length=300, description="the most key points of the content")

def SummaryTemplate(text_article: str, summary_model):
    return [
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data parser.",
                "You will be provided text (which may be in Arabic) and a Pydantic schema.",
                "If the text is not in English, translate it to English first.",
                "Your response MUST be a single valid JSON object following the provided schema, and nothing else.",
                "The summary must be in English and focus on key points and technical or useful information.",
                "Do not include any explanations, markdown, or additional text.",
                "Here is an example of the expected output:",
                '{"Summary": "This is a summary of the content."}',
                "If you are unsure, make your best guess. Only output the JSON."
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## text",
                text_article,
                "",
                "## Pydantic Schema",
                json.dumps(summary_model.model_json_schema(), ensure_ascii=False),
                "",
                "## Summarization (return only JSON):"
            ])
        }
    ]

def extract_first_json(text):
    stack = []
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if not stack:
                start = i
            stack.append(c)
        elif c == '}':
            if stack:
                stack.pop()
                if not stack:
                    json_str = text[start:i+1]
                    return json_str
    raise ValueError("No valid JSON object found")

def SummaryGenerate(message, tokenizer, model, max_new_tokens=1024):
    input_ids = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
    )

    gen_tokens = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(input_ids, gen_tokens)
    ]
    gen_text = tokenizer.decode(gen_tokens[0])
    gen_text = gen_text.replace("<|END_RESPONSE|>", "").replace("<|END_OF_TURN_TOKEN|>", "")
    gen_text = re.sub(r"^```json|^```|```$", "", gen_text, flags=re.MULTILINE).strip()
    json_block = extract_first_json(gen_text)
    return json.loads(json_block)
