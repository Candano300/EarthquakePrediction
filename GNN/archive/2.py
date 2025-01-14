import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error



ratings_df = pd.read_csv('./ratings.csv')
movies_df = pd.read_csv('./movies.csv')
tags_df = pd.read_csv('./tags.csv')

movie_encoder = LabelEncoder()
all_movie_ids = pd.concat([ratings_df['movieId'], movies_df['movieId']]).unique()
movie_encoder.fit(all_movie_ids)

ratings_df['movieId'] = movie_encoder.transform(ratings_df['movieId'])
movies_df['movieId'] = movie_encoder.transform(movies_df['movieId'])
tags_df['movieId'] = movie_encoder.transform(tags_df['movieId'])

user_encoder = LabelEncoder()
ratings_df['userId'] = user_encoder.fit_transform(ratings_df['userId'])
tags_df['userId'] = user_encoder.transform(tags_df['userId'])

num_users = ratings_df['userId'].nunique()
edge_index = torch.tensor(
    np.vstack([
        ratings_df['userId'].values,
        ratings_df['movieId'].values + num_users
    ]), dtype=torch.long
)

edge_weight = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
multi_label_binarizer = MultiLabelBinarizer()
genre_features = multi_label_binarizer.fit_transform(movies_df['genres'])
genre_features = torch.tensor(genre_features, dtype=torch.float)

movie_popularity = ratings_df.groupby('movieId').size().reset_index(name='popularity')
movies_df = pd.merge(movies_df, movie_popularity, on='movieId', how='left')
movies_df['popularity'].fillna(0, inplace=True)

avg_rating = ratings_df.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
movies_df = pd.merge(movies_df, avg_rating, on='movieId', how='left')
movies_df['avg_rating'].fillna(movies_df['avg_rating'].mean(), inplace=True)

movies_df['popularity'] = movies_df['popularity'] / movies_df['popularity'].max()
movies_df['avg_rating'] = movies_df['avg_rating'] / 5.0

popularity_feature = torch.tensor(movies_df['popularity'].values, dtype=torch.float32).unsqueeze(1)
avg_rating_feature = torch.tensor(movies_df['avg_rating'].values, dtype=torch.float32).unsqueeze(1)

genre_features = genre_features.to(dtype=torch.float32)

combined_movie_features = torch.cat([genre_features, popularity_feature, avg_rating_feature], dim=1)

num_movies = movies_df['movieId'].nunique()
user_features = torch.zeros((num_users, combined_movie_features.shape[1]), dtype=torch.float32)
node_features = torch.cat([user_features, combined_movie_features], dim=0)


graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

print(graph_data)

class GCNRecommendationModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GCNRecommendationModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Final layer for prediction
        self.dropout = torch.nn.Dropout(0.2)  # Dropout added

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.fc(x)


input_dim = node_features.shape[1]
hidden_dim = 64
output_dim = 1

model = GCNRecommendationModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
criterion = torch.nn.HuberLoss(delta=0.3)

def train():
    model.train()
    optimizer.zero_grad()
    
    out = model(graph_data.x, graph_data.edge_index)
    
    edge_users = graph_data.edge_index[0]
    edge_movies = graph_data.edge_index[1]

    pred_ratings = (out[edge_users] * out[edge_movies]).sum(dim=1)

    loss = criterion(pred_ratings, graph_data.edge_attr)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()


for epoch in range(1, 200):
    loss = train()
    scheduler.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

def evaluate():
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        edge_users = graph_data.edge_index[0]
        edge_movies = graph_data.edge_index[1]

        pred_ratings = (out[edge_users] * out[edge_movies]).sum(dim=1)

        rmse = mean_squared_error(graph_data.edge_attr.cpu(), pred_ratings.cpu(), squared=False)
        mae = mean_absolute_error(graph_data.edge_attr.cpu(), pred_ratings.cpu())
        
        print(f"\nModel Evaluation:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")

evaluate()
