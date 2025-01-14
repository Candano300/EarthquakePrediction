# GNN Based Movie Recommendation

## Project Overview

The goal of this project was to build a Graph Neural Network (GNN) model to predict movie ratings and recommend movies to users. The primary focus was on understanding how Graph Convolution and Message Passing mechanisms work in GNNs, how inference is conducted, and how edge features are incorporated into the model.
The data is taken from [MovieLens](https://grouplens.org/datasets/movielens/) 
## Project Workflow

### 1. Data Loading and Preprocessing

**Datasets Used:**
- `ratings.csv`: Contains user-movie ratings.
- `movies.csv`: Contains movie metadata (title, genres).
- `tags.csv`: Contains user tags for movies.

**Preprocessing Steps:**
- **Missing Value Handling:** Checked for missing values in all datasets and filled them appropriately (`ratings_df`: 0, `movies_df`: 0, `tags_df`: 0).
- **Label Encoding:** Applied Label Encoding to convert `movieId` and `userId` to numerical indices.
- **Rating Normalization:** Normalized movie ratings to [0, 1] range to stabilize model training.
- **Genre Encoding:** Encoded movie genres as multi-hot vectors using `MultiLabelBinarizer`.
- **Feature Enrichment:** Calculated movie popularity (number of ratings) and average rating, normalized both, and merged with genre features.
- **Edge Construction with RDDs:** Used PySpark RDDs for efficient graph edge construction, improving scalability.

**Graph Construction:**
- **Graph Structure:** Created `edge_index` to represent user-movie interactions.
- **Edge Attributes:** Used normalized ratings as `edge_attr` to weigh edges, ensuring meaningful message passing between nodes.

### 2. Model Architecture

**Initial Model:**
- A 2-layer GCN with ReLU activation predicted ratings via a dot product between user and movie embeddings.

**Enhanced Model:**
- **Three GCN Layers:** Expanded to a 3-layer GCN for deeper learning.
- **Batch Normalization:** Added after each GCN layer to stabilize training.
- **Dropout (0.3):** Introduced for regularization and to prevent overfitting.
- **Optimizer:** Switched to AdamW optimizer with weight decay for better generalization.
- **Learning Rate Scheduler:** Implemented `ReduceLROnPlateau` for adaptive learning rate control.
- **Loss Function:** Changed to `SmoothL1Loss` (Huber Loss) for robustness against outliers.
- **Gradient Clipping:** Applied with `max_norm=1.0` to prevent gradient explosion.
- **Early Stopping:** Stopped training early if validation loss didn't improve.

**Final Model Architecture:**
```
Input -> GCNConv -> BatchNorm -> ReLU -> Dropout
      -> GCNConv -> BatchNorm -> ReLU -> Dropout
      -> GCNConv -> BatchNorm -> ReLU -> Dropout
      -> Fully Connected Layer -> Rating Prediction
```

### 3. Experiments and Observations

**Successful Modifications:**
- **Edge Features Integration:** Incorporating popularity and average ratings improved performance.
- **Batch Normalization:** Enhanced convergence speed and stability.
- **SmoothL1Loss:** Reduced sensitivity to outliers.

**Unsuccessful Modifications:**
- **Larger Hidden Layers:** Increasing hidden dimensions beyond 64 led to overfitting.
- **High Dropout (>0.4):** Hampered learning capacity.
- **Removing AdamW:** Switching back to standard Adam optimizer reduced performance.

### 4. Model Evaluation

**Evaluation Metrics:**
- **Root Mean Squared Error (RMSE):** Measures the average magnitude of prediction errors.
- **Mean Absolute Error (MAE):** Captures the average difference between predictions and actual ratings.

**Final Results:**
- **RMSE:** 0.2296
- **MAE:** 0.1849

Old runs can be found in the archive folder. 

### Reference

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

