import os
import re
import torch
import boto3
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW

def extract_relevant_context(text, keywords_list):
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant_sentences = [s for s in sentences if any(kw.lower() in s.lower() for kw in keywords_list)]
    combined = " ".join(relevant_sentences)
    return combined[:2000]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fine-Tuning for Boundary Cases on device: {device}")
    
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('SupplyChainArticles')
    items = table.scan()['Items']
    df = pd.DataFrame(items)
    
    search_categories = {
        0: ['forced labor', 'child labor', 'trafficking', 'slavery', 'unpaid', 'bondage', 'correctional', 'inmate'],
        1: ['disruption', 'fine', 'probe', 'strike', 'recall', 'redundancy', 'streamlining'],
        2: ['responsibility', 'sustainability', 'earnings', 'partnership', 'mentorship', 'love']
    }
    master_keywords = [phrase for sublist in search_categories.values() for phrase in sublist]
    
    df['CleanText'] = df['Text'].apply(lambda x: extract_relevant_context(str(x), master_keywords))
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    model.to(device)
    
    inputs = tokenizer(df['CleanText'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    labels = torch.tensor(df['Label'].astype(int).values)
    
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=0.01) 
    
    model.train()
    for epoch in range(15):
        total_loss = 0
        for batch in train_loader:
            model.zero_grad()
            b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
            outputs = model(b_input_ids, attention_mask=b_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/15 | Avg Loss: {total_loss/len(train_loader):.4f}")

    output_dir = '/opt/ml/model'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Adversarial Fine-Tuning Complete. Model saved.")

if __name__ == "__main__":
    train()
