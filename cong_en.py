import pandas as pd
import numpy as np
import math
import seaborn 
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

# Nhập các module từ thư viện Surprise
import surprise
import sklearn
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Nhập các module từ sklearn cho phần triển khai thủ công
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# print(pd.__version__)
# print(np.__version__)
# print(surprise.__version__)
# print(sklearn.__version__)

# --- 1. Tìm, Chọn Dữ liệu và Xử lý Dữ liệu ---
# print("--- 1. Tải và Xử lý Dữ liệu (steam-200k.csv) ---")

df = pd.read_csv(r"C:\Users\thanh\Downloads\steam-200k.csv\steam-200k.csv", header=None)
df.columns = ['user_id', 'game_title', 'behavior_name', 'value', 'unused']

# Bỏ cột 'unused'
df = df.drop('unused', axis=1)

# Lọc các hàng chỉ có 'play' behavior
play_df = df[df['behavior_name'] == 'play'].copy()
play_df = play_df.drop('behavior_name', axis=1)
print(play_df.describe())

GAME_PLAY_THRESHOLD = 52

log_val = np.log1p(play_df['value'])
log_threshold = np.log1p(GAME_PLAY_THRESHOLD)

rating = (log_val / log_threshold) * 10
play_df['play_time'] = play_df['value']
play_df['value'] = np.clip(rating, 1, 10)

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

play_df['user_idx'] = user_encoder.fit_transform(play_df['user_id'])
play_df['item_idx'] = item_encoder.fit_transform(play_df['game_title'])

fill_id = play_df.groupby('user_idx').size()
fill_game = play_df.groupby('item_idx').size()

FILL_VAL = 3
GU = fill_id[fill_id >= FILL_VAL].index
GI = fill_game[fill_game >= FILL_VAL].index

filtered = play_df[
	play_df['user_idx'].isin(GU) &
	play_df['item_idx'].isin(GI)
]

# scaler = StandardScaler()
# values = scaler.fit_transform(filtered['value'].values.reshape(-1, 1)).flatten()
values = filtered['value']
# print(pd.DataFrame(values).describe())
# plt.hist(values, bins=100, edgecolor='black')

coo = coo_matrix(
	(values, (filtered['user_idx'], filtered['item_idx'])), 
	shape=(len(user_encoder.classes_), len(item_encoder.classes_))
)


# # Đổi tên cột thành 'user', 'item', 'rating' theo chuẩn của các hệ thống đề xuất
play_df.rename(columns={'user_id': 'user', 'game_title': 'item', 'value': 'rating'}, inplace=True)

coo_csr = coo.tocsr()
user_similarity_matrix = cosine_similarity(coo_csr)

user_sim_df_manual = pd.DataFrame(user_similarity_matrix)

# print(play_df.describe())
print(play_df.head(2))
print("5 dòng đầu của DataFrame Độ tương đồng Người dùng (KNN thủ công):")
print(user_sim_df_manual.head())
print("\n")

# # --- 2. Triển khai: Sử dụng thư viện Surprise ---
# print("--- 2. Triển khai sử dụng thư viện Surprise ---")

# # Tạo đối tượng Reader cho Surprise Dataset
# # rating_scale phải khớp với min/max của cột 'rating' đã biến đổi 
# sur_dat = play_df.copy()
# print("surprise", sur_dat.describe())

# ss = StandardScaler()

# # 1, khong thuc hien scale va dung scale tu bien doi sim ben tren # Loi 32%
# sur_dat['rating'] = np.log1p(sur_dat['play_time']) # su dung log1p # Loi 16%
# # sur_dat['rating'] = ss.fit_transform(sur_dat[['play_time']]) # su dung standard # Loi 2%

# min_rating_surprise = sur_dat['rating'].min()
# max_rating_surprise = sur_dat['rating'].max()
# print(f"Khoảng giá trị của rating sau biến đổi: Nhỏ nhất={min_rating_surprise:.4f}, Lớn nhất={max_rating_surprise:.4f}")
# reader = Reader(rating_scale=(min_rating_surprise, max_rating_surprise))

# # Tải dữ liệu từ play_df vào một đối tượng Surprise Dataset
# data_surprise = Dataset.load_from_df(sur_dat[['user', 'item', 'rating']], reader)

# # Chia tập dữ liệu thành trainset và testset
# # SỬ DỤNG surprise_train_test_split ĐÃ ĐỔI TÊN
# trainset_surprise, testset_surprise = surprise_train_test_split(data_surprise, test_size=.25, random_state=42) # Sử dụng 25% làm tập kiểm tra

# # print(f"Tổng số {trainset_surprise.n_users} người dùng và {trainset_surprise.n_items} mục trong tập huấn luyện của Surprise\n")

# # TASK: Thực hiện lọc cộng tác dựa trên KNN trên ma trận tương tác người dùng-mục
# # TODO: Huấn luyện mô hình lọc cộng tác dựa trên KNN bằng trainset và đánh giá kết quả bằng testset:

# # - Định nghĩa một mô hình KNNBasic()
# # Thử các kết hợp hyperparameter khác nhau để xem cái nào có hiệu suất tốt nhất
# # Ví dụ: user_based = True (Lọc cộng tác dựa trên người dùng)
# print("Huấn luyện mô hình KNNBasic dựa trên người dùng với Surprise...")
# algo_user_based_surprise = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
# # - Huấn luyện mô hình KNNBasic trên trainset
# algo_user_based_surprise.fit(trainset_surprise)
# # - Dự đoán rating cho testset
# predictions_user_based_surprise = algo_user_based_surprise.test(testset_surprise)
# # - Sau đó tính RMSE
# print("RMSE của KNNBasic dựa trên người dùng (Surprise):")
# rmse = accuracy.rmse(predictions_user_based_surprise, verbose=True)
# print(f"Độ lỗi của mô hình: {rmse/(max_rating_surprise - min_rating_surprise):.4f}\n")


# --- 3. Implementation: Manual KNN Collaborative Filtering ---

# Step 1: Split the filtered data into training and testing sets
# We use the 'filtered' DataFrame as it contains the processed data with user/item indices
train_df, test_df = sklearn_train_test_split(filtered, test_size=0.2, random_state=42)

# Get the total number of unique users and items from the original encoders
# This ensures the dimensions of the matrices are consistent with the full encoding space
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)

print(f"\n--- Debugging Dimensions ---")
print(f"Total unique users from user_encoder: {num_users}")
print(f"Total unique items from item_encoder: {num_items}")
print(f"Max user_idx in train_df: {train_df['user_idx'].max()}")
print(f"Max item_idx in train_df: {train_df['item_idx'].max()}")
print(f"Min user_idx in train_df: {train_df['user_idx'].min()}")
print(f"Min item_idx in train_df: {train_df['item_idx'].min()}")
print(f"--- Debugging Dimensions End ---\n")

# Step 2: Create a sparse user-item matrix from the training data
# Using csr_matrix for efficient row slicing, which is beneficial for similarity calculations
train_user_item_matrix = csr_matrix(
    (train_df['value'], (train_df['user_idx'], train_df['item_idx'])),
    shape=(num_users, num_items)
)

# Step 3: Calculate user similarity matrix using cosine similarity on the training matrix
# This matrix will be used to find the k nearest neighbors for prediction
user_similarity_matrix_train = cosine_similarity(train_user_item_matrix)

print("\n5 dòng đầu của DataFrame Độ tương đồng Người dùng (KNN thủ công - trên tập huấn luyện):")
print(pd.DataFrame(user_similarity_matrix_train).head())
print("\n")


# Calculate average rating for each user from the training data
# Get sums of ratings and counts of non-zero ratings per user
sum_ratings_per_user = np.array(train_user_item_matrix.sum(axis=1)).flatten()
count_ratings_per_user = np.array((train_user_item_matrix != 0).sum(axis=1)).flatten()

# Avoid division by zero by setting counts to 1 where they are 0
# This makes the division safe, and nan_to_num will handle the rest
count_ratings_per_user[count_ratings_per_user == 0] = 1

user_avg_ratings = sum_ratings_per_user / count_ratings_per_user
user_avg_ratings = np.nan_to_num(user_avg_ratings, nan=0.0) # Ensure it's float and NaNs become 0.0


def predict_rating_knn_manual(user_idx, item_idx, k,
                              user_sim_matrix, train_matrix, avg_ratings):
    """
    Estimates the rating for a given user-item pair using User-Based KNN Collaborative Filtering.

    Args:
        user_idx (int): The index of the user for whom to predict the rating.
        item_idx (int): The index of the item for which to predict the rating.
        k (int): The number of nearest neighbors to consider.
        user_sim_matrix (np.array): The user-user similarity matrix derived from training data.
        train_matrix (scipy.sparse.csr_matrix): The user-item interaction matrix from training data.
        avg_ratings (np.array): Array of average ratings for each user from training data.

    Returns:
        float: The predicted rating for the user-item pair.
               Returns the user's average rating if no suitable neighbors are found.
               Returns the global average rating if the user also has no average rating.
    """
    # Calculate global average rating as a fallback for cold users/items
    global_avg_rating = train_matrix.sum() / (train_matrix != 0).sum()
    global_avg_rating = np.nan_to_num(global_avg_rating, nan=5.0) # Default to 5.0 if no ratings at all

    # Check if user_idx or item_idx are out of bounds for the matrices
    if user_idx >= user_sim_matrix.shape[0] or item_idx >= train_matrix.shape[1]:
        return global_avg_rating # Return global average for out-of-bounds indices

    user_similarities = user_sim_matrix[user_idx]

    # Exclude the user themselves and sort by similarity in descending order
    # Get indices of users sorted by similarity (excluding self)
    similar_users_indices = user_similarities.argsort()[::-1]
    similar_users_indices = similar_users_indices[similar_users_indices != user_idx]

    weighted_sum = 0
    sum_abs_similarities = 0
    neighbors_found = 0

    # Get the target user's average rating.
    # Ensure it's a scalar value.
    user_avg_rating_target = avg_ratings[user_idx] if user_idx < len(avg_ratings) else 0.0
    if isinstance(user_avg_rating_target, np.ndarray): # Check if it's a NumPy array
        user_avg_rating_target = user_avg_rating_target.item() # Convert 0-dim array to scalar

    # Fallback if the target user has no average rating (e.g., new user in training set after filtering)
    if user_avg_rating_target == 0.0: # Explicitly compare with 0.0 (float)
        return global_avg_rating


    # Iterate through similar users to find k nearest neighbors who rated the item
    for neighbor_idx in similar_users_indices:
        if neighbors_found >= k:
            break

        # Check if the neighbor_idx is within the bounds of the training matrix and avg_ratings
        if neighbor_idx >= train_matrix.shape[0] or neighbor_idx >= len(avg_ratings):
            continue

        # Check if the neighbor has rated the target item (item_idx)
        # train_matrix[neighbor_idx, item_idx] returns a sparse matrix with one element if rated, else 0
        neighbor_rating = train_matrix[neighbor_idx, item_idx]
        if neighbor_rating > 0: # If the rating exists (non-zero in sparse matrix)
            similarity = user_similarities[neighbor_idx]
            
            # Get neighbor's average rating and convert to scalar if necessary
            neighbor_avg_rating = avg_ratings[neighbor_idx]
            if isinstance(neighbor_avg_rating, np.ndarray): # Check if it's a NumPy array
                neighbor_avg_rating = neighbor_avg_rating.item() # Convert 0-dim array to scalar

            # Ensure neighbor_avg_rating is not zero to avoid division by zero or incorrect calculation
            if neighbor_avg_rating == 0.0: # Explicitly compare with 0.0 (float)
                continue

            weighted_sum += similarity * (neighbor_rating - neighbor_avg_rating)
            sum_abs_similarities += abs(similarity)
            neighbors_found += 1

    # Apply the KNN collaborative filtering formula: R_ui = avg(R_u) + sum(sim(u,v) * (R_vi - avg(R_v))) / sum(|sim(u,v)|)
    if sum_abs_similarities > 0:
        predicted_rating = user_avg_rating_target + (weighted_sum / sum_abs_similarities)
        # Clip the predicted rating to be within the valid range (1 to 10)
        return np.clip(predicted_rating, 1, 10)
    else:
        # If no suitable neighbors are found who rated the item,
        # return the target user's average rating.
        # If the target user also has no average rating, return the global average.
        return user_avg_rating_target if user_avg_rating_target > 0.0 else global_avg_rating

# Set the number of neighbors (k) to consider for prediction
k_neighbors = 20 # This value can be tuned for better performance

# Initialize lists to store actual and predicted ratings for RMSE calculation
actual_ratings = []
predicted_ratings = []

# Iterate over the test set to make predictions for each user-item pair
print(f"Bắt đầu dự đoán cho {len(test_df)} mục trong tập dữ liệu kiểm tra...")
for index, row in test_df.iterrows():
    user_idx = int(row['user_idx'])
    item_idx = int(row['item_idx'])
    actual_rating = row['value'] # Use 'value' from the filtered df, not 'rating' from play_df

    # Ensure user_idx and item_idx are within the bounds of the matrices
    # This handles cases where a user/item might be in the test set but not in the training set
    # (though 'filtered' data should minimize this, it's good for robustness).
    if user_idx < num_users and item_idx < num_items:
        predicted_rating = predict_rating_knn_manual(
            user_idx, item_idx, k_neighbors,
            user_similarity_matrix_train, train_user_item_matrix, user_avg_ratings
        )
        actual_ratings.append(actual_rating)
        predicted_ratings.append(predicted_rating)
    else:
        # This case should be less frequent with the updated num_users/num_items calculation,
        # but it's good to keep for robustness if test_df contains indices not seen by encoders.
        # For now, we'll just skip these to avoid errors.
        # print(f"Skipping prediction for out-of-bounds indices: user_idx={user_idx}, item_idx={item_idx}")
        pass

# Step 5: Calculate RMSE (Root Mean Squared Error)
if len(actual_ratings) > 0:
    rmse_manual = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    print(f"\nRMSE (KNN thủ công) = {rmse_manual:.4f}")
    print(f"Độ lỗi của mô hình: {rmse_manual/10:.4f}\n")
else:
    print("\nKhông có đủ dữ liệu trong tập kiểm tra để tính RMSE. Vui lòng kiểm tra tập dữ liệu hoặc quá trình xử lý.")

print("\nHoàn tất triển khai KNN thủ công.")
