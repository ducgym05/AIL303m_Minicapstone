import pandas as pd
import numpy as np
import math
import seaborn
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

# Import modules from Surprise library
import surprise
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Import modules from sklearn for manual implementation
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# --- 1. Data Loading and Preprocessing ---
print("--- 1. Tải và Xử lý Dữ liệu (steam-200k.csv) ---")

# Try to load the CSV. If not found, create dummy data for demonstration.
try:
    df = pd.read_csv("steam-200k.csv", header=None)
except FileNotFoundError:
    print("steam-200k.csv không tìm thấy. Đang tạo dữ liệu giả để minh họa.")
    data = {
        0: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        1: ['GameA', 'GameB', 'GameA', 'GameC', 'GameB', 'GameC', 'GameD', 'GameA', 'GameB', 'GameD', 'GameE', 'GameF', 'GameE', 'GameF', 'GameE', 'GameA', 'GameB', 'GameC', 'GameD', 'GameE', 'GameG', 'GameH', 'GameG', 'GameI', 'GameH', 'GameI', 'GameJ', 'GameG', 'GameH', 'GameJ'],
        2: ['play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'buy', 'buy', 'buy', 'buy', 'buy', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play', 'play'],
        3: [100, 50, 120, 30, 80, 60, 200, 150, 90, 110, 70, 40, 130, 180, 20, 0, 0, 0, 0, 0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)

df.columns = ['user_id', 'game_title', 'behavior_name', 'value', 'unused']

# Drop 'unused' column
df = df.drop('unused', axis=1)

# Filter rows with 'play' behavior only
play_df = df[df['behavior_name'] == 'play'].copy()
play_df = play_df.drop('behavior_name', axis=1)

GAME_PLAY_THRESHOLD = 52

log_val = np.log1p(play_df['value'])
log_threshold = np.log1p(GAME_PLAY_THRESHOLD)

rating = (log_val / log_threshold) * 10
play_df['play_time'] = play_df['value'] # Keep original play time
play_df['value'] = np.clip(rating, 1, 10) # Transformed rating, clipped to 1-10

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

play_df['user_idx'] = user_encoder.fit_transform(play_df['user_id'])
play_df['item_idx'] = item_encoder.fit_transform(play_df['game_title'])

fill_id = play_df.groupby('user_idx').size()
fill_game = play_df.groupby('item_idx').size()

FILL_VAL = 3 # Minimum interactions for a user/item to be included
GU = fill_id[fill_id >= FILL_VAL].index
GI = fill_game[fill_game >= FILL_VAL].index

# Filter data to include only users and items that meet the minimum interaction threshold
filtered = play_df[
    play_df['user_idx'].isin(GU) &
    play_df['item_idx'].isin(GI)
].copy() # Use .copy() to avoid SettingWithCopyWarning

num_users_total = len(user_encoder.classes_)
num_items_total = len(item_encoder.classes_)

print(f"Tổng số người dùng duy nhất: {num_users_total}")
print(f"Tổng số mục duy nhất: {num_items_total}")
print(f"Kích thước DataFrame đã lọc: {filtered.shape}")
print(f"5 dòng đầu của DataFrame đã lọc:\n{filtered.head()}")


# --- Class for Surprise Implementation (Item-Based) ---
class SurpriseRecommender:
    def __init__(self, data_df, user_encoder, item_encoder, k_neighbors=20):
        """
        Initializes the SurpriseRecommender with data and trains the KNNBasic model for item-based recommendations.

        Args:
            data_df (pd.DataFrame): DataFrame with 'user_id', 'game_title', 'value' (transformed rating).
            user_encoder (LabelEncoder): Fitted LabelEncoder for users.
            item_encoder (LabelEncoder): Fitted LabelEncoder for items.
            k_neighbors (int): Number of neighbors for KNN.
        """
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.k_neighbors = k_neighbors

        # Prepare data for Surprise: rename columns to 'user', 'item', 'rating'
        sur_dat = data_df.rename(columns={'user_id': 'user', 'game_title': 'item', 'value': 'rating'}).copy()

        # Determine rating scale for Surprise Reader
        min_rating_surprise = sur_dat['rating'].min()
        max_rating_surprise = sur_dat['rating'].max()
        print(f"\nKhoảng giá trị của rating cho Surprise: Nhỏ nhất={min_rating_surprise:.4f}, Lớn nhất={max_rating_surprise:.4f}")

        reader = Reader(rating_scale=(min_rating_surprise, max_rating_surprise))
        self.data_surprise = Dataset.load_from_df(sur_dat[['user', 'item', 'rating']], reader)

        # Build full trainset from the entire dataset for similarity calculations
        self.trainset_surprise = self.data_surprise.build_full_trainset()
        
        print(f"Tổng số {self.trainset_surprise.n_users} người dùng và {self.trainset_surprise.n_items} mục trong tập huấn luyện của Surprise\n")

        # Initialize and train KNNBasic model for ITEM-BASED recommendations
        # user_based=False means item-based collaborative filtering
        print("Huấn luyện mô hình KNNBasic dựa trên MỤC với Surprise...")
        self.algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}, k=self.k_neighbors)
        self.algo.fit(self.trainset_surprise)
        print("Huấn luyện mô hình Surprise hoàn tất.")

        # Optional: Evaluate RMSE on a test set if needed for performance comparison
        # You can uncomment and use this block if you want to see the RMSE for Surprise
        # trainset_eval, testset_eval = surprise_train_test_split(self.data_surprise, test_size=0.25, random_state=42)
        # self.algo.fit(trainset_eval)
        # predictions = self.algo.test(testset_eval)
        # rmse = accuracy.rmse(predictions, verbose=False)
        # print(f"RMSE của KNNBasic dựa trên mục (Surprise) trên tập kiểm tra: {rmse:.4f}")
        # print(f"Độ lỗi của mô hình: {rmse/(max_rating_surprise - min_rating_surprise):.4f}\n")


    def recommend(self, game_idx, num_recommendations=6):
        """
        Recommends similar games based on item-based collaborative filtering using the Surprise library.

        Args:
            game_idx (int): The integer-encoded index of the game to find recommendations for.
            num_recommendations (int): The number of similar games to recommend.

        Returns:
            list: A list of integer-encoded game_idx values for the recommended games.
                  Returns an empty list if the game_idx is not found or no recommendations can be made.
        """
        try:
            # Convert external game_idx to Surprise's internal item ID
            # First, get the original game title from the item_encoder
            original_game_title = self.item_encoder.inverse_transform([game_idx])[0]
            # Then, get Surprise's internal item ID (iid) from the original game title
            inner_item_id = self.algo.trainset.to_inner_iid(original_game_title)

            # Get the k most similar items (neighbors) to the target item
            # self.algo.get_neighbors returns a list of inner item IDs
            # We fetch more than needed to filter out the input item itself
            similar_inner_item_ids = self.algo.get_neighbors(inner_item_id, k=self.k_neighbors + num_recommendations) 

            recommended_game_indices = []
            for inner_id in similar_inner_item_ids:
                # Convert Surprise's internal item ID back to original game title
                original_title = self.algo.trainset.to_raw_iid(inner_id)
                # Convert original game title back to integer-encoded game_idx
                encoded_game_idx = self.item_encoder.transform([original_title])[0]

                if encoded_game_idx != game_idx: # Exclude the input game itself
                    recommended_game_indices.append(encoded_game_idx)
                
                if len(recommended_game_indices) >= num_recommendations:
                    break

            return recommended_game_indices[:num_recommendations]

        except ValueError as e:
            print(f"Lỗi: Không tìm thấy game_idx {game_idx} trong dữ liệu huấn luyện của Surprise hoặc lỗi chuyển đổi: {e}")
            return []
        except Exception as e:
            print(f"Lỗi không xác định khi đề xuất với Surprise: {e}")
            return []


# --- Class for Manual Scikit-learn Implementation (User-Based for RMSE, Item-Based for Recommendation) ---
class ScikitLearnRecommender:
    def __init__(self, filtered_df, user_encoder, item_encoder, num_users, num_items, k_neighbors=20):
        """
        Initializes the ScikitLearnRecommender with processed data and prepares matrices.

        Args:
            filtered_df (pd.DataFrame): DataFrame with 'user_idx', 'item_idx', 'value' (transformed rating).
            user_encoder (LabelEncoder): Fitted LabelEncoder for users.
            item_encoder (LabelEncoder): Fitted LabelEncoder for items.
            num_users (int): Total number of unique users.
            num_items (int): Total number of unique items.
            k_neighbors (int): Number of neighbors for KNN.
        """
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.num_users = num_users
        self.num_items = num_items
        self.k_neighbors = k_neighbors

        # Split data into training and testing sets for RMSE evaluation
        self.train_df, self.test_df = sklearn_train_test_split(filtered_df, test_size=0.2, random_state=42)

        # Create sparse user-item matrix from the training data
        self.train_user_item_matrix = csr_matrix(
            (self.train_df['value'], (self.train_df['user_idx'], self.train_df['item_idx'])),
            shape=(self.num_users, self.num_items)
        )

        # Calculate user similarity matrix (for user-based prediction, used in RMSE evaluation)
        self.user_similarity_matrix_train = cosine_similarity(self.train_user_item_matrix)

        # Calculate item similarity matrix (for item-based recommendations)
        # Transpose the user-item matrix to get item-user matrix for item similarity
        self.item_similarity_matrix_train = cosine_similarity(self.train_user_item_matrix.T)
        print("\n5 dòng đầu của DataFrame Độ tương đồng MỤC (KNN thủ công - trên tập huấn luyện):")
        print(pd.DataFrame(self.item_similarity_matrix_train).head())
        print("\n")

        # Calculate average rating for each user from the training data
        sum_ratings_per_user = np.array(self.train_user_item_matrix.sum(axis=1)).flatten()
        count_ratings_per_user = np.array((self.train_user_item_matrix != 0).sum(axis=1)).flatten()
        count_ratings_per_user[count_ratings_per_user == 0] = 1 # Avoid division by zero
        self.user_avg_ratings = sum_ratings_per_user / count_ratings_per_user
        self.user_avg_ratings = np.nan_to_num(self.user_avg_ratings, nan=0.0) # Ensure NaNs become 0.0

        # Calculate global average rating as a fallback
        self.global_avg_rating = self.train_user_item_matrix.sum() / (self.train_user_item_matrix != 0).sum()
        self.global_avg_rating = np.nan_to_num(self.global_avg_rating, nan=5.0) # Default to 5.0 if no ratings

        print("Huấn luyện mô hình Scikit-learn thủ công hoàn tất.")

        # Evaluate RMSE on test set using the user-based prediction logic
        self._evaluate_rmse()


    def _evaluate_rmse(self):
        """
        Evaluates the RMSE of the manual KNN model on the test set using user-based prediction.
        This is an internal helper method.
        """
        actual_ratings = []
        predicted_ratings = []

        print(f"Bắt đầu dự đoán cho {len(self.test_df)} mục trong tập dữ liệu kiểm tra (Scikit-learn thủ công)...")
        for index, row in self.test_df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            actual_rating = row['value']

            # Ensure user_idx and item_idx are within the bounds of the matrices
            if user_idx < self.num_users and item_idx < self.num_items:
                predicted_rating = self._predict_rating_knn_manual(
                    user_idx, item_idx, self.k_neighbors,
                    self.user_similarity_matrix_train, self.train_user_item_matrix, self.user_avg_ratings
                )
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)
            else:
                # This case should be rare with proper data filtering
                pass

        if len(actual_ratings) > 0:
            rmse_manual = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            print(f"\nRMSE (KNN thủ công Scikit-learn) = {rmse_manual:.4f}")
            print(f"Độ lỗi của mô hình: {rmse_manual/10:.4f}\n")
        else:
            print("\nKhông có đủ dữ liệu trong tập kiểm tra để tính RMSE cho Scikit-learn thủ công.")


    def _predict_rating_knn_manual(self, user_idx, item_idx, k,
                                   user_sim_matrix, train_matrix, avg_ratings):
        """
        Estimates the rating for a given user-item pair using User-Based KNN Collaborative Filtering.
        (Helper method, internal to the class)
        """
        # Return global average for out-of-bounds indices
        if user_idx >= user_sim_matrix.shape[0] or item_idx >= train_matrix.shape[1]:
            return self.global_avg_rating

        user_similarities = user_sim_matrix[user_idx]

        # Get indices of users sorted by similarity (excluding self)
        similar_users_indices = user_similarities.argsort()[::-1]
        similar_users_indices = similar_users_indices[similar_users_indices != user_idx]

        weighted_sum = 0
        sum_abs_similarities = 0
        neighbors_found = 0

        # Get the target user's average rating.
        user_avg_rating_target = avg_ratings[user_idx] if user_idx < len(avg_ratings) else 0.0
        if isinstance(user_avg_rating_target, np.ndarray):
            user_avg_rating_target = user_avg_rating_target.item()

        # Fallback if the target user has no average rating
        if user_avg_rating_target == 0.0:
            return self.global_avg_rating

        # Iterate through similar users to find k nearest neighbors who rated the item
        for neighbor_idx in similar_users_indices:
            if neighbors_found >= k:
                break

            # Check if neighbor_idx is within bounds
            if neighbor_idx >= train_matrix.shape[0] or neighbor_idx >= len(avg_ratings):
                continue

            # Check if the neighbor has rated the target item
            neighbor_rating = train_matrix[neighbor_idx, item_idx]
            if neighbor_rating > 0: # If the rating exists (non-zero in sparse matrix)
                similarity = user_similarities[neighbor_idx]
                
                neighbor_avg_rating = avg_ratings[neighbor_idx]
                if isinstance(neighbor_avg_rating, np.ndarray):
                    neighbor_avg_rating = neighbor_avg_rating.item()

                if neighbor_avg_rating == 0.0:
                    continue

                weighted_sum += similarity * (neighbor_rating - neighbor_avg_rating)
                sum_abs_similarities += abs(similarity)
                neighbors_found += 1

        # Apply the KNN collaborative filtering formula
        if sum_abs_similarities > 0:
            predicted_rating = user_avg_rating_target + (weighted_sum / sum_abs_similarities)
            return np.clip(predicted_rating, 1, 10) # Clip to valid range
        else:
            # If no suitable neighbors are found, return user's average or global average
            return user_avg_rating_target if user_avg_rating_target > 0.0 else self.global_avg_rating


    def recommend(self, game_idx, num_recommendations=6):
        """
        Recommends similar games based on item-item collaborative filtering using the manually calculated
        cosine similarity matrix.

        Args:
            game_idx (int): The integer-encoded index of the game to find recommendations for.
            num_recommendations (int): The number of similar games to recommend.

        Returns:
            list: A list of integer-encoded game_idx values for the recommended games.
                  Returns an empty list if the game_idx is not found or no recommendations can be made.
        """
        if game_idx >= self.num_items or game_idx < 0:
            print(f"Lỗi: game_idx {game_idx} nằm ngoài phạm vi các mục đã biết.")
            return []

        # Get similarities for the target item from the item similarity matrix
        item_similarities = self.item_similarity_matrix_train[game_idx]

        # Sort items by similarity in descending order, excluding the item itself
        similar_items_indices = item_similarities.argsort()[::-1]
        
        recommended_game_indices = []
        count = 0
        for idx in similar_items_indices:
            if idx == game_idx: # Skip the item itself
                continue
            
            # Ensure the item index is valid and similarity is not NaN
            if idx < self.num_items and not np.isnan(item_similarities[idx]):
                recommended_game_indices.append(idx)
                count += 1
            
            if count >= num_recommendations:
                break
        
        return recommended_game_indices


# --- Main Execution ---
if __name__ == "__main__":
    # Initialize Surprise Recommender
    print("\n--- Khởi tạo và chạy Surprise Recommender ---")
    # We pass the original play_df (before filtering by FILL_VAL) to Surprise
    # because Surprise handles its own internal filtering/mapping.
    # The 'value' column in play_df is the transformed rating.
    surprise_rec = SurpriseRecommender(play_df, user_encoder, item_encoder)

    # Example recommendation for Surprise
    # Pick a game_idx that exists in the filtered data for a meaningful example
    if not filtered.empty:
        example_game_idx_surprise = filtered['item_idx'].iloc[0] # Take the first game_idx from filtered data
        original_game_title_surprise = item_encoder.inverse_transform([example_game_idx_surprise])[0]
        print(f"\nĐề xuất cho game_idx (Surprise): {example_game_idx_surprise} (Game: '{original_game_title_surprise}')")
        
        recommended_games_surprise = surprise_rec.recommend(example_game_idx_surprise, num_recommendations=6)
        
        if recommended_games_surprise:
            print(f"6 game_idx được đề xuất bởi Surprise: {recommended_games_surprise}")
            # Convert back to original game titles for better understanding
            recommended_titles_surprise = item_encoder.inverse_transform(recommended_games_surprise)
            print(f"Tên game được đề xuất bởi Surprise: {list(recommended_titles_surprise)}")
        else:
            print("Không có đề xuất nào từ Surprise.")
    else:
        print("Không có đủ dữ liệu đã lọc để chạy ví dụ Surprise.")


    # # Initialize Scikit-learn Recommender
    # print("\n--- Khởi tạo và chạy Scikit-learn Recommender ---")
    # # We pass the 'filtered' DataFrame to ScikitLearnRecommender as it uses the pre-filtered indices.
    # sklearn_rec = ScikitLearnRecommender(filtered, user_encoder, item_encoder, num_users_total, num_items_total)

    # # Example recommendation for Scikit-learn
    # if not filtered.empty:
    #     example_game_idx_sklearn = filtered['item_idx'].iloc[1] # Take another game_idx
    #     original_game_title_sklearn = item_encoder.inverse_transform([example_game_idx_sklearn])[0]
    #     print(f"\nĐề xuất cho game_idx (Scikit-learn): {example_game_idx_sklearn} (Game: '{original_game_title_sklearn}')")
        
    #     recommended_games_sklearn = sklearn_rec.recommend(example_game_idx_sklearn, num_recommendations=6)

    #     if recommended_games_sklearn:
    #         print(f"6 game_idx được đề xuất bởi Scikit-learn: {recommended_games_sklearn}")
    #         # Convert back to original game titles for better understanding
    #         recommended_titles_sklearn = item_encoder.inverse_transform(recommended_games_sklearn)
    #         print(f"Tên game được đề xuất bởi Scikit-learn: {list(recommended_titles_sklearn)}")
    #     else:
    #         print("Không có đề xuất nào từ Scikit-learn.")
    # else:
    #     print("Không có đủ dữ liệu đã lọc để chạy ví dụ Scikit-learn.")
