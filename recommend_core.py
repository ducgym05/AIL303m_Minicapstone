import pandas as pd
import numpy as np
import math
import seaborn
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
import re
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
# --- LỚP NÀY ĐƯỢC GIỮ NGUYÊN HOÀN TOÀN ---
class SurpriseRecommender:
    def __init__(self, data_df, user_encoder, item_encoder, k_neighbors=20):
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.k_neighbors = k_neighbors
        sur_dat = data_df.rename(columns={'user_id': 'user', 'game_title': 'item', 'value': 'rating'}).copy()
        min_rating_surprise = sur_dat['rating'].min()
        max_rating_surprise = sur_dat['rating'].max()
        print(f"\nKhoảng giá trị của rating cho Surprise: Nhỏ nhất={min_rating_surprise:.4f}, Lớn nhất={max_rating_surprise:.4f}")
        reader = Reader(rating_scale=(min_rating_surprise, max_rating_surprise))
        self.data_surprise = Dataset.load_from_df(sur_dat[['user', 'item', 'rating']], reader)
        self.trainset_surprise = self.data_surprise.build_full_trainset()
        print(f"Tổng số {self.trainset_surprise.n_users} người dùng và {self.trainset_surprise.n_items} mục trong tập huấn luyện của Surprise\n")
        print("Huấn luyện mô hình KNNBasic dựa trên MỤC với Surprise...")
        self.algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}, k=self.k_neighbors)
        self.algo.fit(self.trainset_surprise)
        print("Huấn luyện mô hình Surprise hoàn tất.")

    def recommend(self, game_idx, num_recommendations=6):
        try:
            original_game_title = self.item_encoder.inverse_transform([game_idx])[0]
            inner_item_id = self.algo.trainset.to_inner_iid(original_game_title)
            similar_inner_item_ids = self.algo.get_neighbors(inner_item_id, k=self.k_neighbors + num_recommendations) 
            recommended_game_indices = []
            for inner_id in similar_inner_item_ids:
                original_title = self.algo.trainset.to_raw_iid(inner_id)
                encoded_game_idx = self.item_encoder.transform([original_title])[0]
                if encoded_game_idx != game_idx:
                    recommended_game_indices.append(encoded_game_idx)
                if len(recommended_game_indices) >= num_recommendations:
                    break
            return recommended_game_indices[:num_recommendations]
        except ValueError as e:
            return []
        except Exception as e:
            return []

# --- Class for Manual Scikit-learn Implementation ---
# --- LỚP NÀY ĐƯỢC GIỮ NGUYÊN HOÀN TOÀN ---
class ScikitLearnRecommender:
    def __init__(self, filtered_df, user_encoder, item_encoder, num_users, num_items, k_neighbors=20):
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.num_users = num_users
        self.num_items = num_items
        self.k_neighbors = k_neighbors
        self.train_df, self.test_df = sklearn_train_test_split(filtered_df, test_size=0.2, random_state=42)
        self.train_user_item_matrix = csr_matrix(
            (self.train_df['value'], (self.train_df['user_idx'], self.train_df['item_idx'])),
            shape=(self.num_users, self.num_items)
        )
        self.user_similarity_matrix_train = cosine_similarity(self.train_user_item_matrix)
        self.item_similarity_matrix_train = cosine_similarity(self.train_user_item_matrix.T)
        print("\n5 dòng đầu của DataFrame Độ tương đồng MỤC (KNN thủ công - trên tập huấn luyện):")
        print(pd.DataFrame(self.item_similarity_matrix_train).head())
        print("\n")
        sum_ratings_per_user = np.array(self.train_user_item_matrix.sum(axis=1)).flatten()
        count_ratings_per_user = np.array((self.train_user_item_matrix != 0).sum(axis=1)).flatten()
        count_ratings_per_user[count_ratings_per_user == 0] = 1 # Avoid division by zero
        self.user_avg_ratings = sum_ratings_per_user / count_ratings_per_user
        self.user_avg_ratings = np.nan_to_num(self.user_avg_ratings, nan=0.0) # Ensure NaNs become 0.0
        self.global_avg_rating = self.train_user_item_matrix.sum() / (self.train_user_item_matrix != 0).sum()
        self.global_avg_rating = np.nan_to_num(self.global_avg_rating, nan=5.0) # Default to 5.0 if no ratings
        print("Huấn luyện mô hình Scikit-learn thủ công hoàn tất.")
        self._evaluate_rmse()
    def _evaluate_rmse(self):
        actual_ratings = []
        predicted_ratings = []
        print(f"Bắt đầu dự đoán cho {len(self.test_df)} mục trong tập dữ liệu kiểm tra (Scikit-learn thủ công)...")
        for index, row in self.test_df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            actual_rating = row['value']
            if user_idx < self.num_users and item_idx < self.num_items:
                predicted_rating = self._predict_rating_knn_manual(
                    user_idx, item_idx, self.k_neighbors,
                    self.user_similarity_matrix_train, self.train_user_item_matrix, self.user_avg_ratings
                )
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)
        if len(actual_ratings) > 0:
            rmse_manual = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            print(f"\nRMSE (KNN thủ công Scikit-learn) = {rmse_manual:.4f}")
            print(f"Độ lỗi của mô hình: {rmse_manual/10:.4f}\n")
        else:
            print("\nKhông có đủ dữ liệu trong tập kiểm tra để tính RMSE cho Scikit-learn thủ công.")
    def _predict_rating_knn_manual(self, user_idx, item_idx, k, user_sim_matrix, train_matrix, avg_ratings):
        if user_idx >= user_sim_matrix.shape[0] or item_idx >= train_matrix.shape[1]:
            return self.global_avg_rating
        user_similarities = user_sim_matrix[user_idx]
        similar_users_indices = user_similarities.argsort()[::-1]
        similar_users_indices = similar_users_indices[similar_users_indices != user_idx]
        weighted_sum = 0
        sum_abs_similarities = 0
        neighbors_found = 0
        user_avg_rating_target = avg_ratings[user_idx] if user_idx < len(avg_ratings) else 0.0
        if isinstance(user_avg_rating_target, np.ndarray):
            user_avg_rating_target = user_avg_rating_target.item()
        if user_avg_rating_target == 0.0:
            return self.global_avg_rating
        for neighbor_idx in similar_users_indices:
            if neighbors_found >= k:
                break
            if neighbor_idx >= train_matrix.shape[0] or neighbor_idx >= len(avg_ratings):
                continue
            neighbor_rating = train_matrix[neighbor_idx, item_idx]
            if neighbor_rating > 0:
                similarity = user_similarities[neighbor_idx]
                neighbor_avg_rating = avg_ratings[neighbor_idx]
                if isinstance(neighbor_avg_rating, np.ndarray):
                    neighbor_avg_rating = neighbor_avg_rating.item()
                if neighbor_avg_rating == 0.0:
                    continue
                weighted_sum += similarity * (neighbor_rating - neighbor_avg_rating)
                sum_abs_similarities += abs(similarity)
                neighbors_found += 1
        if sum_abs_similarities > 0:
            predicted_rating = user_avg_rating_target + (weighted_sum / sum_abs_similarities)
            return np.clip(predicted_rating, 1, 10)
        else:
            return user_avg_rating_target if user_avg_rating_target > 0.0 else self.global_avg_rating
    def recommend(self, game_idx, num_recommendations=6):
        if game_idx >= self.num_items or game_idx < 0:
            print(f"Lỗi: game_idx {game_idx} nằm ngoài phạm vi các mục đã biết.")
            return []
        item_similarities = self.item_similarity_matrix_train[game_idx]
        similar_items_indices = item_similarities.argsort()[::-1]
        recommended_game_indices = []
        count = 0
        for idx in similar_items_indices:
            if idx == game_idx:
                continue
            if idx < self.num_items and not np.isnan(item_similarities[idx]):
                recommended_game_indices.append(idx)
                count += 1
            if count >= num_recommendations:
                break
        return recommended_game_indices


# =========================================================================
# === PHẦN BỔ SUNG: KHỞI TẠO CÁC BIẾN VÀ HÀM CẦN THIẾT CHO WEB APP
# =========================================================================

print("\n--- [WEB INIT] Chuẩn bị các chức năng cho Web App ---")

# Tải và chuẩn bị dữ liệu chi tiết game
try:
    df_details = pd.read_csv("game_details_with_image_links.csv", encoding='latin1', engine='python', header=None)
    df_details.columns = ['title', 'description', 'image_link']
    print("[INFO WEB] Tải thành công game_details_with_image_links.csv.")
except Exception as e:
    print(f"[WARNING WEB] Không thể đọc file chi tiết game: {e}.")
    df_details = pd.DataFrame(columns=['title', 'description', 'image_link'])

def normalize_title_for_url(title):
    """Hàm này chỉ dùng để tạo URL thân thiện, không ảnh hưởng đến logic gốc."""
    if not isinstance(title, str): return ""
    text = title.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text).strip()
    return text

# Tạo cột tên chuẩn hóa cho URL trong cả hai dataframe
df['normalized_title_url'] = df['game_title'].apply(normalize_title_for_url)
df_details['normalized_title_url'] = df_details['title'].apply(normalize_title_for_url)

# Tạo bảng tra cứu tổng thể để kết hợp thông tin (không dùng fuzzywuzzy)
game_lookup_web = pd.merge(
    df[['game_title', 'normalized_title_url']].drop_duplicates(),
    df_details.drop_duplicates(subset=['normalized_title_url']),
    on='normalized_title_url',
    how='left'
)
game_lookup_web['display_title'] = game_lookup_web['title'].fillna(game_lookup_web['game_title'])
game_lookup_web.set_index('normalized_title_url', inplace=True)

# Khởi tạo một đối tượng Recommender mới dành riêng cho web
# Việc này đảm bảo không can thiệp vào các đối tượng trong phần code gốc của bạn
print("\n--- [WEB INIT] Khởi tạo Recommender và các hàm chức năng cho Web ---")

# Khởi tạo một đối tượng Recommender mới dành riêng cho web
surprise_rec_web = SurpriseRecommender(play_df, user_encoder, item_encoder)

def get_game_info(normalized_url):
    try:
        info = game_lookup_web.loc[normalized_url]
        original_title = info.get('display_title')
        description = info.get('description', "Không có mô tả chi tiết.")
        image_path = info.get('image_link', "")
        if pd.isna(original_title): original_title = normalized_url.replace('-', ' ').title()
        if pd.isna(description): description = "Không có mô tả chi tiết."
        if pd.isna(image_path): image_path = ""
        return original_title, description, image_path
    except KeyError:
        return normalized_url.replace('-', ' ').title(), "Không có thông tin chi tiết.", ""

def get_recommendations(normalized_url, n=6):
    try:
        original_title = game_lookup_web.loc[normalized_url, 'game_title']
        game_idx = item_encoder.transform([original_title])[0]
        recommended_indices = surprise_rec_web.recommend(game_idx, num_recommendations=n)
        recommended_titles = item_encoder.inverse_transform(recommended_indices)
        return [normalize_title_for_url(t) for t in recommended_titles]
    except (KeyError, ValueError, IndexError):
        return []

def get_game_rating(normalized_url):
    try:
        original_title = game_lookup_web.loc[normalized_url, 'game_title']
        avg_rating = play_df[play_df['game_title'] == original_title]['value'].mean()
        return f"{avg_rating:.1f}" if pd.notna(avg_rating) else "3.5"
    except KeyError:
        return "3.5"

def get_top_rated_games(num_games=10):
    top_games_df = play_df.groupby('game_title')['value'].mean().nlargest(num_games)
    return [normalize_title_for_url(title) for title in top_games_df.index]

def get_random_games_from_top(num_games=20, top_n=200):
    top_games_df = play_df.groupby('game_title')['value'].mean().nlargest(top_n)
    if top_games_df.empty:
        return []
    sample_titles = top_games_df.sample(min(num_games, len(top_games_df))).index.tolist()
    return [normalize_title_for_url(title) for title in sample_titles]

def search_games(query, n=20):
    if not query: return []
    search_term = query.lower()
    matches = game_lookup_web[game_lookup_web['display_title'].str.lower().str.contains(search_term, na=False)]
    return matches.index.tolist()[:n]

# --- Main Execution ---
# --- PHẦN NÀY ĐƯỢC GIỮ NGUYÊN HOÀN TOÀN ĐỂ BẠN KIỂM TRA NHƯ CŨ ---
if __name__ == "__main__":
    # Initialize Surprise Recommender
    print("\n--- Khởi tạo và chạy Surprise Recommender (Test) ---")
    surprise_rec_test = SurpriseRecommender(play_df, user_encoder, item_encoder)

    # Example recommendation for Surprise
    if not filtered.empty:
        example_game_idx_surprise = filtered['item_idx'].iloc[0]
        original_game_title_surprise = item_encoder.inverse_transform([example_game_idx_surprise])[0]
        print(f"\nĐề xuất cho game_idx (Surprise): {example_game_idx_surprise} (Game: '{original_game_title_surprise}')")
        
        recommended_games_surprise = surprise_rec_test.recommend(example_game_idx_surprise, num_recommendations=6)
        
        if recommended_games_surprise:
            print(f"6 game_idx được đề xuất bởi Surprise: {recommended_games_surprise}")
            recommended_titles_surprise = item_encoder.inverse_transform(recommended_games_surprise)
            print(f"Tên game được đề xuất bởi Surprise: {list(recommended_titles_surprise)}")
        else:
            print("Không có đề xuất nào từ Surprise.")
    else:
        print("Không có đủ dữ liệu đã lọc để chạy ví dụ Surprise.")