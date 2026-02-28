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
    
    p_red = probs[0].item()
    p_yellow = probs[1].item()
    p_green = probs[2].item()

    risk_score = (p_red * 100) + (p_yellow * 50) + (p_green * 0)
    
    if risk_score >= 70:
        status = "RED"
    elif risk_score >= 35:
        status = "YELLOW"
    else:
        status = "GREEN"
        
    return {
        "risk_score": round(risk_score, 2),
        "status": status,
        "breakdown": {
            "critical_risk": round(p_red, 4),
            "moderate_risk": round(p_yellow, 4),
            "safe": round(p_green, 4)
        }
    }

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.dumps(json.loads(request_body).get('headline', ''))
    return request_body

def output_fn(prediction, response_content_type):
    return json.dumps(prediction)