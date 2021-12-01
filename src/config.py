import os
from pathlib import Path
from dataclasses import dataclass

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from fastT5 import (
    export_and_get_onnx_model,
    generate_onnx_representation,
    quantize,
    get_onnx_model,
    get_onnx_runtime_sessions,
    OnnxT5,
)

@dataclass(frozen=True)
class Config:
    MODEL_DIR: str = "./model/BeIR-query-gen-msmarco-t5-large-v1"


def onnx_model_init(qg_model):
    
    encoder_path = os.path.join(qg_model,f"{Path(qg_model).stem}-encoder-quantized.onnx")
    decoder_path = os.path.join(qg_model,f"{Path(qg_model).stem}-decoder-quantized.onnx")
    init_decoder_path = os.path.join(qg_model, f"{Path(qg_model).stem}-init-decoder-quantized.onnx")
    tokenizer_path = qg_model
    
    model_paths = encoder_path, decoder_path, init_decoder_path

    model_sessions = get_onnx_runtime_sessions(model_paths)
    model = OnnxT5(qg_model, model_sessions)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer


def generate_questions(user_utterance, model, tokenizer):
    
    input_ids = tokenizer.encode(user_utterance, return_tensors="pt")
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=3,
    )
    
    queries =[]
    for i in range(len(outputs)):
        queries.append(tokenizer.decode(outputs[i], skip_special_tokens=True))
        
    return queries