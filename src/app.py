import json
from config import Config, onnx_model_init, generate_questions

conf = Config()
qg_model, qg_tokenizer = onnx_model_init(conf.MODEL_DIR)

def lambda_handler(event, context):
    
    header = {"Content-Type": "application/json"}
    payload = {"message": "Lambda container image invoked!", "event": event}
    
    user_utterance = event["user_utterance"]
    
    queries = generate_questions(user_utterance, qg_model, qg_tokenizer)

    return {
        "user_utterance": user_utterance,
        "no_of_queries": len(queries),
        "generated_queries": queries,
        "statusCode": 200,
    }