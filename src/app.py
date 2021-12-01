import json
from config import Config, onnx_model_init, generate_question

conf = Config()
qg_model, qg_tokenizer = onnx_model_init(conf.MODEL_DIR)

def lambda_handler(event, context):
    
    header = {"Content-Type": "application/json"}
    payload = {"message": "Lambda container image invoked!", "event": event}
    
    user_utterance = event["user_utterance"]
    
    query = generate_question(user_utterance, qg_model, qg_tokenizer)

    return {
        "header": header,
        "body": json.dumps(payload),
        "query": json.dumps(query),
        "statusCode": 200,
    }