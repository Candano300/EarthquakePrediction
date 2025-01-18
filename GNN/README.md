# GNN-Based Movie Recommendation

## Project Overview

This project focuses on developing a **Graph Neural Network (GNN)** model to predict movie ratings and recommend movies to users. The model leverages **Graph Convolution** and **Message Passing** mechanisms, enhancing the recommendation process by incorporating user-movie interactions and enriched movie features.

**Dataset:** [MovieLens](https://grouplens.org/datasets/movielens/)

## Project Workflow

### 1. Data Loading and Preprocessing

**Datasets Used:**
- `ratings.csv`: User-movie ratings.
- `movies.csv`: Movie metadata (title, genres).
- `tags.csv`: User-assigned tags for movies.

**Preprocessing Steps:**
- **Missing Value Handling:** Filled missing values (`ratings_df`: 0, `movies_df`: 'Unknown', `tags_df`: 'Unknown').
- **Label Encoding:** Encoded `movieId` and `userId` for numerical processing.
- **Genre Encoding:** Multi-hot encoded genres using `MultiLabelBinarizer`.
- **Feature Enrichment:** Integrated **movie popularity** (number of ratings) and **average rating** into movie features.
- **Data Splitting:** Split the data into **70% Training**, **15% Validation**, and **15% Test** sets.
- **Graph Construction:** Built `edge_index` for user-movie interactions and used **ratings as edge attributes** (`edge_attr`).

### 2. Model Architecture

**Model Design:**
- **3-Layer GCN:** Deeper learning with three GCN layers for user-movie relationships.
- **Batch Normalization:** Stabilizes training after each GCN layer.
- **Dropout (0.2/0.3):** Prevents overfitting.
- **Optimizer:** **AdamW** optimizer with weight decay for better generalization.
- **Learning Rate Scheduler:** **ReduceLROnPlateau** to adapt learning rates dynamically.
- **Loss Function:** **SmoothL1Loss** for robustness against outliers.
- **Gradient Clipping:** Applied with `max_norm=1.0`.
- **Early Stopping:** Stops training if the validation loss doesn't improve over 10 epochs.

**Final Model Architecture:**
```
Input -> GCNConv -> BatchNorm -> ReLU -> Dropout
      -> GCNConv -> BatchNorm -> ReLU -> Dropout
      -> GCNConv -> BatchNorm -> ReLU -> Dropout
      -> Fully Connected Layer -> Rating Prediction
```

### 3. Hyperparameter Optimization

**Grid Search Parameters:**
- **Hidden Dimensions:** `[64, 128]`
- **Dropout Rates:** `[0.2, 0.3]`
- **Learning Rates:** `[0.005, 0.001]`
- **Weight Decays:** `[1e-4, 1e-3]`

**Optimization Strategy:**
- Iterates through all hyperparameter combinations.
- Uses **validation RMSE** for model selection.
- Implements **early stopping** and **adaptive learning rate scheduling**.

### 4. Model Evaluation

**Evaluation Metrics:**
- **Root Mean Squared Error (RMSE):** Measures prediction error magnitude.
- **Mean Absolute Error (MAE):** Captures the average prediction error.

**Final Results:**
- **RMSE:** 1.1216  
- **MAE:** 0.9039

### 5. Observations

**Improvements:**
- **Feature Integration:** Combining **genre**, **popularity**, and **average rating** improved accuracy.
- **Batch Normalization** 
- **Grid Search** 

**Unsuccessful Attempts:**
- **Larger Hidden Layers (>128):** Led to overfitting.
- **High Dropout (>0.4):** Reduced learning capacity.

### 6. Conclusion

The enhanced GNN model with **hyperparameter optimization**, **early stopping**, and enriched movie features successfully predicts movie ratings with high accuracy. The integration of additional features and a deeper architecture notably improved performance.

### Reference

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

