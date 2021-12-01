import json
from config import Config


def lambda_handler(event, context):
    # TODO implementation
    header = {"Content-Type": "application/json"}
    payload = {"message": "Lambda container image invoked!", "event": event}

    return {
        "header": header,
        "statusCode": 200,
        "body": json.dumps(payload),
    }