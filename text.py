import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # Add positional encoding to the input embeddings
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class TextClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', hidden_size=768):
        super(TextClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.positional_encoding = PositionalEncoding(d_model=hidden_size, max_len=512)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 128)  # Output layer for binary classification
        # self.sigmoid = nn.Sigmoid()


    def forward(self, text_tensor):
        """
        Forward method for the TextClassifier.
        
        :param text_tensor: A concatenated tensor containing input_ids and attention_mask.
        :return: Binary classification output.
        """
        # Split the concatenated tensor back into input_ids and attention_mask
        input_ids = text_tensor[:, :512]  # First half is input_ids
        attention_mask = text_tensor[:, 512:]  # Second half is attention_mask

        # Ensure tensors are on the same device as the model
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)

        # Get BERT embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Apply positional encoding
        encoded_states = self.positional_encoding(hidden_states)

        # Use [CLS] token for classification
        cls_token = encoded_states[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Feed through fully connected layer
        x = self.dropout(cls_token)
        x = self.fc(x)

        # print('Passed the Tensors from text.py file ')
        return x

    

    # def forward(self, text):
    #     # Tokenize input text
    #     encoded = self.tokenizer(
    #         text,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt",
    #         max_length=512
    #     )
    #     input_ids = encoded['input_ids'].to(self.bert.device)
    #     attention_mask = encoded['attention_mask'].to(self.bert.device)

    #     # Get BERT embeddings
    #     outputs = self.bert(input_ids, attention_mask=attention_mask)
    #     hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

    #     # Apply positional encoding
    #     encoded_states = self.positional_encoding(hidden_states)

    #     # Use [CLS] token for classification
    #     cls_token = encoded_states[:, 0, :]  # Shape: (batch_size, hidden_size)

    #     # Feed through fully connected layer
    #     x = self.dropout(cls_token)
    #     x = self.fc(x)
    #     x = self.sigmoid(x)  # Output in range [0, 1]
    #     print('Passed the Tensors from text.py file ')
    #     return x