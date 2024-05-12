from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

class ModelPredictor:
    def __init__(self, model_path, labels_path, tokenizer):
        self.model_path = model_path
        self.labels_path = labels_path
        self.tokenizer = tokenizer
        self.model = self.initialize_model()
        self.label_encoder = self.initialize_label_encoder()

    def read_labels_file(self, file_path):
        with open(file_path, 'r') as file:
            labels_str = file.read()
        labels_list = [str(label.strip()) for label in labels_str.split(',') if label.strip()]
        return labels_list

    def initialize_model(self):
        labels = self.read_labels_file(self.labels_path)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))
        state_dict = torch.load(self.model_path)
        model.load_state_dict(state_dict)
        return model

    def initialize_label_encoder(self):
        labels = self.read_labels_file(self.labels_path)
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        return label_encoder
    
    def make_prediction(self, text, top_n=1, type='softmax'):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if type == 'softmax':
            probabilities = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
        else:
            probabilities = torch.sigmoid(outputs.logits).squeeze(0).cpu().numpy()

        predicted_classes = np.argsort(-probabilities)[:top_n]
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)

        return predicted_labels, probabilities[predicted_classes]


class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.df = pd.read_csv(data_file, sep='\t', header=0, names=['id', 'title', 'text', 'keywords', 'correct']).fillna('')

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_id = self.df.loc[idx, 'id']
        title = self.df.loc[idx, 'title']
        text = self.df.loc[idx, 'text']
        keywords = self.df.loc[idx, 'keywords']
        return {
            'text_id': text_id,
            'title': title,
            'text': text,
            'keywords': keywords
        }


def format_data(data, threshold, normalization='not'):
    formatted_string = ""
    seen_ids = set()
    filtered_data = []

    for row in data:
        if row[0] not in seen_ids:
            seen_ids.add(row[0])
            prob = float(row[1])
            if prob > threshold:
                filtered_data.append((row[0], prob))

    total_prob = sum(prob for _, prob in filtered_data)

    if filtered_data:
        filtered_data = sorted(filtered_data, key=lambda x: x[1], reverse=True)
        for id, prob in filtered_data:
            if normalization == 'all' or (normalization == 'some' and total_prob > 1):
                prob = prob / total_prob

            prob = round(prob, 4)
            formatted_string += f"{id}-{prob}\\"
    else:
        formatted_string = 'EMPTY\\'
    return formatted_string.rstrip("\\")

def load_dictionary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return eval(file.read())
     