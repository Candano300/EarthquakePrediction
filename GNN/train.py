import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn import BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from itertools import product

# Load datasets
ratings_df = pd.read_csv('./ratings.csv')
movies_df = pd.read_csv('./movies.csv')
tags_df = pd.read_csv('./tags.csv')

# Handle missing values
ratings_df.fillna(0, inplace=True)
movies_df.fillna('Unknown', inplace=True)
tags_df.fillna('Unknown', inplace=True)

# Encode movieId and userId
movie_encoder = LabelEncoder()
all_movie_ids = pd.concat([ratings_df['movieId'], movies_df['movieId']]).unique()
movie_encoder.fit(all_movie_ids)
ratings_df['movieId'] = movie_encoder.transform(ratings_df['movieId'])
movies_df['movieId'] = movie_encoder.transform(movies_df['movieId'])
tags_df['movieId'] = movie_encoder.transform(tags_df['movieId'])

user_encoder = LabelEncoder()
ratings_df['userId'] = user_encoder.fit_transform(ratings_df['userId'])
tags_df['userId'] = user_encoder.transform(tags_df['userId'])

# Split data into 70% train, 15% validation, 15% test
train_ratings, temp_ratings = train_test_split(ratings_df, test_size=0.3, random_state=42)
val_ratings, test_ratings = train_test_split(temp_ratings, test_size=0.5, random_state=42)

num_users = ratings_df['userId'].nunique()
num_movies = movies_df['movieId'].nunique()

# Helper function to create edge index and weights
def create_edge_index(data):
    edge_index = torch.tensor(np.vstack([data['userId'].values, data['movieId'].values + num_users]), dtype=torch.long)
    edge_weight = torch.tensor(data['rating'].values, dtype=torch.float)
    return edge_index, edge_weight

edge_index_train, edge_weight_train = create_edge_index(train_ratings)
edge_index_val, edge_weight_val = create_edge_index(val_ratings)
edge_index_test, edge_weight_test = create_edge_index(test_ratings)

movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
multi_label_binarizer = MultiLabelBinarizer()
genre_features = multi_label_binarizer.fit_transform(movies_df['genres'])
genre_features = torch.tensor(genre_features, dtype=torch.float32)

# Combine user and movie features
user_features = torch.zeros((num_users, genre_features.shape[1]), dtype=torch.float32)
node_features = torch.cat([user_features, genre_features], dim=0)

# Define the GCN model
class GCNRecommendationModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GCNRecommendationModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)
        return self.fc(x)

# Grid Search Parameters
hidden_dims = [64, 128]
dropouts = [0.2, 0.3]
learning_rates = [0.005, 0.001]
weight_decays = [1e-4, 1e-3]

best_model = None
best_rmse = float('inf')
best_params = {}

# Grid Search
for hidden_dim, dropout, lr, wd in product(hidden_dims, dropouts, learning_rates, weight_decays):
    print(f"Training with hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}, weight_decay={wd}")
    model = GCNRecommendationModel(node_features.shape[1], hidden_dim, 1, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = torch.nn.SmoothL1Loss()
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(node_features, edge_index_train)
        pred_ratings = (out[edge_index_train[0]] * out[edge_index_train[1]]).sum(dim=1)
        loss = criterion(pred_ratings, edge_weight_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(node_features, edge_index_val)
            pred_val = (out_val[edge_index_val[0]] * out_val[edge_index_val[1]]).sum(dim=1)
            val_loss = criterion(pred_val, edge_weight_val)

        #print(f"Epoch {epoch}: Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
        
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping triggered.")
            break

    rmse = mean_squared_error(edge_weight_val.cpu(), pred_val.cpu(), squared=False)
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model.state_dict()
        best_params = {'hidden_dim': hidden_dim, 'dropout': dropout, 'lr': lr, 'weight_decay': wd}
        print(f"New best model found with RMSE: {best_rmse:.4f}")

# Final Evaluation on Test Set
print("\nFinal evaluation on the test set with best hyperparameters...")
model = GCNRecommendationModel(node_features.shape[1], best_params['hidden_dim'], 1, best_params['dropout'])
model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    out_test = model(node_features, edge_index_test)
    pred_test = (out_test[edge_index_test[0]] * out_test[edge_index_test[1]]).sum(dim=1)
    rmse_test = mean_squared_error(edge_weight_test.cpu(), pred_test.cpu(), squared=False)
    mae_test = mean_absolute_error(edge_weight_test.cpu(), pred_test.cpu())
    print(f"\nTest Set Evaluation:\nRMSE: {rmse_test:.4f}\nMAE: {mae_test:.4f}")
