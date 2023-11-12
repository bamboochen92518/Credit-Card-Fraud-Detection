from myfunc import ReadCSV, Sort_Data_with_CANO_and_Time, Tokenize_and_Normalize_Data, Data_to_Dataset
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from accelerate import Accelerator

accelerator = Accelerator()

# Paremeters
csv_file_path = '../final/dataset_1st/training.csv'
numeric_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
category_features = ['chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']
num_numeric_features = len(numeric_features)
hidden_size = 16
window_size = 8
output_size = 1
learning_rate = 0.01
train_batch = 32
do_shuffle = True
train_epoch = 5
output_dir = '1111'
vocab_file = f'{output_dir}/vocab.json'

data = ReadCSV(csv_file_path)
tokenized_data, vocab_size = Tokenize_and_Normalize_Data(data, numeric_features, category_features, vocab_file)
sorted_data = Sort_Data_with_CANO_and_Time(tokenized_data, window_size) # list(list(dictionary * window size) * window)
vector_size = list()
for v in vocab_size:
    vector_size.append(min(16, v))

class MyRNNModel(nn.Module):
    def __init__(self, num_numeric_features, vocab_size, vector_size, hidden_size, output_size, window_size):
        super(MyRNNModel, self).__init__()

        # Embedding 层
        self.embeddings = nn.ModuleList()
        for num_categories, embedding_dim in zip(vocab_size, vector_size):
            try:
                embedding_layer = nn.Embedding(num_categories, embedding_dim)
                self.embeddings.append(embedding_layer)
            except IndexError as e:
                print(f"Error: {e}, num_categories: {num_categories}")
                raise e
        # LSTM 层
        self.rnn = nn.LSTM(input_size=num_numeric_features + sum(vector_size), hidden_size=hidden_size, batch_first=True)

        # 全连接层
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, numeric_data, category_data):
        numeric_data = numeric_data.tolist()
        batch_size = len(numeric_data)
        final_numeric_data = list()
        for w in range(window_size):
            split_numeric_data = list()
            for b in range(batch_size):
                split_numeric_data.append(numeric_data[b][w*(num_numeric_features):(w+1)*(num_numeric_features)])
            split_numeric_data = torch.tensor(split_numeric_data).cuda()
            final_numeric_data.append(split_numeric_data)
        num_categories = int(len(category_data) / window_size)
        category_data = [category_data[i:i+num_categories] for i in range(0, len(category_data), num_categories)]
       
        combined_features = final_numeric_data[0].unsqueeze(1)
        numeric_data = list()
        for i in range(window_size):
            # Embedding
            embedded_categories = []
            for embedding, category in zip(self.embeddings, category_data[i]):
                try:
                    embedded_category = embedding(category)
                    embedded_categories.append(embedded_category)
                except IndexError as e:
                    print(f"Error: {e}, embedding_dim: {embedding.embedding_dim}, num_categories: {embedding.num_embeddings}")
                    print("category:", category)
                    raise e
            # print("Embedding Category: ")
            # print(embedded_categories)
            # [tensor([[[vector size]]], grad_fn=...)] * (category num + 1)
            embedded_categories_combined = torch.cat(embedded_categories, dim=-1)
            # print("Embedding Category Combined: ")
            # print(embedded_categories_combined)
            # [tensor([[[sum(vector size)]]])]
            
            # Concatenate numeric and embedded category features
            combined_features = torch.cat([final_numeric_data[i].unsqueeze(1), embedded_categories_combined], dim=-1)
            numeric_data.append(combined_features)
        
        combined_features = torch.cat(numeric_data, dim=1)
        # print("Combined Feature: ")
        # print(combined_features)
        # [tensor[[[sum(vector size) + numeric num]]]]

        # RNN forward pass
        rnn_output, _ = self.rnn(combined_features)
        # print(f"Shape of rnn_output: {rnn_output.shape}")

        # Fully connected layer for prediction
        output = self.fc_output(rnn_output[:, -1, :])
        output = self.relu(output)

        return output


model = MyRNNModel(num_numeric_features, vocab_size, vector_size, hidden_size, output_size, window_size)
dataset = Data_to_Dataset(sorted_data, numeric_features, category_features, window_size)
dataloader = DataLoader(dataset, batch_size=train_batch, shuffle=do_shuffle)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
#model = model.to(device)

'''
for x in dataloader:
    print(x)
    break
'''
# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Assuming binary classification
optimizer = Adam(model.parameters(), lr=learning_rate)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Training loop with tqdm
num_epochs = train_epoch
for epoch in range(num_epochs):
    tqdm_dataloader = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True)
    for numeric_data, *category_data, labels in tqdm_dataloader:
        #numeric_data = numeric_data.to(device)
        #category_data = [category.to(device) for category in category_data]
        #labels = labels.to(device)
        optimizer.zero_grad()
        output = model(numeric_data, category_data)
        loss = criterion(output, labels.float())
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        tqdm_dataloader.set_postfix({'Loss': loss.item()}, refresh=True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f'{output_dir}/epoch{epoch + 1}'
    torch.save(model.state_dict(), filename)
