from myfunc import ReadCSV, Sort_Data_with_CANO_and_Time, Tokenize_Data, Data_to_Dataset, Calculate_Embedding_Dimension
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from accelerate import Accelerator

accelerator = Accelerator()

csv_file_path = '../final/dataset_1st/training.csv'
# csv_file_path = '../final/dataset_1st/small_train.csv'
numeric_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
category_features = ['chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'ecfg', 'insfg', 'bnsfg', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'flg_3dsmk']

data = ReadCSV(csv_file_path)
sorted_data = Sort_Data_with_CANO_and_Time(data)
tokenized_data = Tokenize_Data(sorted_data, numeric_features, category_features)

class MyRNNModel(nn.Module):
    def __init__(self, num_numeric_features, vocab_size, vector_size, hidden_size, output_size):
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
        # Embedding
        embedded_categories = []
        for embedding, category in zip(self.embeddings, category_data):
            try:
                embedded_category = embedding(category)
                embedded_categories.append(embedded_category)
            except IndexError as e:
                print(f"Error: {e}, embedding_dim: {embedding.embedding_dim}, num_categories: {embedding.num_embeddings}")
                print("category:", category)
                raise e
        embedded_categories_combined = torch.cat(embedded_categories, dim=-1)

        # Concatenate numeric and embedded category features
        numeric_data = numeric_data.unsqueeze(1)
        combined_features = torch.cat([numeric_data, embedded_categories_combined], dim=-1)

        # RNN forward pass
        rnn_output, _ = self.rnn(combined_features)
        # print(f"Shape of rnn_output: {rnn_output.shape}")

        # Fully connected layer for prediction
        output = self.fc_output(rnn_output[:, -1, :])
        output = self.relu(output)

        return output

# Example usage
num_numeric_features = len(numeric_features)
hidden_size = 16
output_size = 1
learning_rate = 0.01
train_batch = 64
vocab_size = Calculate_Embedding_Dimension(tokenized_data, category_features)
vector_size = list()
for v in vocab_size:
    vector_size.append(min(16, v))
do_shuffle = False
train_epoch = 10
output_dir = '1111'

model = MyRNNModel(num_numeric_features, vocab_size, vector_size, hidden_size, output_size)
dataset = Data_to_Dataset(tokenized_data, numeric_features, category_features)
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
