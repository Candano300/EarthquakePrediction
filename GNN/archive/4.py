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


# Load datasets
ratings_df = pd.read_csv('./ratings.csv')
movies_df = pd.read_csv('./movies.csv')
tags_df = pd.read_csv('./tags.csv')

# Check for missing values
def check_missing(df, name):
    missing = df.isnull().sum()
    print(f"\nMissing values in {name}:")
    print(missing)

check_missing(ratings_df, 'ratings')
check_missing(movies_df, 'movies')
check_missing(tags_df, 'tags')

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

# Normalize ratings for edge weights
edge_weight = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())

# Correct the number of users and movies
num_users = ratings_df['userId'].nunique()
num_movies = movies_df['movieId'].nunique()

# Total number of nodes: users + movies
total_nodes = num_users + num_movies


# Encode genres as multi-hot vectors
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
multi_label_binarizer = MultiLabelBinarizer()
genre_features = multi_label_binarizer.fit_transform(movies_df['genres'])
genre_features = torch.tensor(genre_features, dtype=torch.float32)

# Calculate movie popularity (number of ratings)
movie_popularity = ratings_df.groupby('movieId').size().reset_index(name='popularity')
movies_df = pd.merge(movies_df, movie_popularity, on='movieId', how='left')
movies_df['popularity'].fillna(0, inplace=True)

# Calculate average movie rating
avg_rating = ratings_df.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
movies_df = pd.merge(movies_df, avg_rating, on='movieId', how='left')
movies_df['avg_rating'].fillna(movies_df['avg_rating'].mean(), inplace=True)

# Normalize popularity and average rating
movies_df['popularity'] = movies_df['popularity'] / movies_df['popularity'].max()
movies_df['avg_rating'] = movies_df['avg_rating'] / 5.0

# Convert popularity and avg_rating to tensors
popularity_feature = torch.tensor(movies_df['popularity'].values, dtype=torch.float32).unsqueeze(1)
avg_rating_feature = torch.tensor(movies_df['avg_rating'].values, dtype=torch.float32).unsqueeze(1)

# Combine movie features (genres, popularity, avg_rating)
combined_movie_features = torch.cat([genre_features, popularity_feature, avg_rating_feature], dim=1)

# Create user features as zero vectors
user_features = torch.zeros((num_users, combined_movie_features.shape[1]), dtype=torch.float32)

# Combine user and movie features
node_features = torch.cat([user_features, combined_movie_features], dim=0)

# Correctly build the edge index
edge_index = torch.tensor(
    np.vstack([
        ratings_df['userId'].values,
        ratings_df['movieId'].values + num_users  # Shift movie IDs to avoid overlap with user IDs
    ]), dtype=torch.long
)

# Construct the PyTorch Geometric Data object
graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

print(f"Node Features Shape: {node_features.shape}")
print(f"Graph Data: {graph_data}")




# Define the GNN model with BatchNorm and 3 GCN layers
class GCNRecommendationModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GCNRecommendationModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(self.bn3(x))
        x = self.dropout(x)
        return self.fc(x)

# Initialize model, optimizer, scheduler, and loss function
input_dim = node_features.shape[1]
hidden_dim = 128
output_dim = 1
model = GCNRecommendationModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = torch.nn.SmoothL1Loss()

# Training function with gradient clipping and early stopping
def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    edge_users = graph_data.edge_index[0]
    edge_movies = graph_data.edge_index[1]
    pred_ratings = (out[edge_users] * out[edge_movies]).sum(dim=1)
    loss = criterion(pred_ratings, graph_data.edge_attr)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

# Training loop with early stopping
best_loss = float('inf')
patience_counter = 0
for epoch in range(1, 300):
    loss = train()
    scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    if patience_counter > 10:
        print("Early stopping.")
        break
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Evaluation function
def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        edge_users = graph_data.edge_index[0]
        edge_movies = graph_data.edge_index[1]
        pred_ratings = (out[edge_users] * out[edge_movies]).sum(dim=1)
        rmse = mean_squared_error(graph_data.edge_attr.cpu(), pred_ratings.cpu(), squared=False)
        mae = mean_absolute_error(graph_data.edge_attr.cpu(), pred_ratings.cpu())
        print(f"\nModel Evaluation:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")

evaluate()


