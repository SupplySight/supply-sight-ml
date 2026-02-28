import os
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def model_fn(model_dir):
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def predict_fn(input_data, model_artifacts):
    model, tokenizer = model_artifacts
    inputs = tokenizer(input_data, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    risk_score = (probs[0].item() * 100) + (probs[1].item() * 50)
    if risk_score >= 75: status = "RED"
    elif risk_score >= 40: status = "YELLOW"
    else: status = "GREEN"
    return {"risk_score": risk_score, "status": status}

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.loads(request_body)['headline']
    return request_body

def output_fn(prediction, response_content_type):
    return json.dumps(prediction)
