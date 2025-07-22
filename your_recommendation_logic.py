import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import re # Import module regex để làm sạch chuỗi

# Base URL cho ảnh header của game trên Steam CDN
STEAM_IMAGE_BASE_URL = "https://cdn.akamai.steamstatic.com/steam/apps/"
STEAM_IMAGE_SUFFIX = "/header.jpg" # Hoặc "/hero_capsule.jpg" tùy loại ảnh bạn muốn

# Hàm chuẩn hóa tên game
def normalize_game_title(title):
    if pd.isna(title):
        return None
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title
# Hàm cắt ngắn văn bản
def truncate_text(text, word_limit):
    if not isinstance(text, str):
        return str(text), str(text) # Đảm bảo là chuỗi
    words = text.split()
    if len(words) > word_limit:
        truncated = " ".join(words[:word_limit]) + "..."
        return truncated, text
    return text, text # Trả về cả hai giống nhau nếu không cần cắt
# 1. Tải và xử lý dữ liệu play_df (steam-200k.csv)
df = pd.read_csv('steam-200k.csv', header=None)
df.columns = ['user', 'item', 'behavior', 'value', 'unused']
df = df.drop('unused', axis=1)

play_df = df[df['behavior'] == 'play'].copy()
play_df['rating'] = np.log1p(play_df['value']) # Log transform 'value' (hours played) to 'rating'

# ÁP DỤNG CHUẨN HÓA TÊN GAME CHO play_df
play_df['item_normalized'] = play_df['item'].apply(normalize_game_title)
# Loại bỏ các hàng có tên game bị rỗng sau khi chuẩn hóa
play_df.dropna(subset=['item_normalized'], inplace=True)


# Xác định min và max rating từ dữ liệu đã xử lý để scale
MIN_OBSERVED_RATING = play_df['rating'].min()
MAX_OBSERVED_RATING = play_df['rating'].max()

print(f"MIN_OBSERVED_RATING (log1p): {MIN_OBSERVED_RATING}")
print(f"MAX_OBSERVED_RATING (log1p): {MAX_OBSERVED_RATING}")

def scale_rating(original_rating, min_old=MIN_OBSERVED_RATING, max_old=MAX_OBSERVED_RATING, min_new=1, max_new=10):
    """Scales a rating from its original range to a new range (e.g., 1-10)."""
    if max_old == min_old:
        return (min_new + max_new) / 2
    clamped_rating = max(min_old, min(original_rating, max_old))
    scaled_value = min_new + (clamped_rating - min_old) * (max_new - min_new) / (max_old - min_old)
    return round(scaled_value, 2)

# Tạo ma trận người dùng-item (dựa trên tên game đã chuẩn hóa)
user_item_matrix = play_df.pivot_table(index='user', columns='item_normalized', values='rating', aggfunc='sum').fillna(0)

# Tính toán độ tương đồng giữa các người dùng
user_similarity = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Lấy danh sách tất cả các game (đã chuẩn hóa)
all_games_normalized = user_item_matrix.columns.tolist()
# Tính toán rating trung bình cho mỗi game (dựa trên tên game đã chuẩn hóa)
game_average_ratings_normalized = play_df.groupby('item_normalized')['rating'].mean()


# Tải dữ liệu chi tiết game (game_details_with_image_links.csv/.xlsx)
game_details_df = None
file_path = 'game_details_with_image_links.csv' # Tên file của bạn
try:
    game_details_df = pd.read_excel(file_path)
    print("Attempting to load game_details_with_image_links as Excel (XLSX).")
except Exception as excel_err:
    print(f"Failed to load as Excel: {excel_err}. Trying as CSV...")
    encodings_to_try = ['utf-8', 'latin1', 'cp1252']
    delimiters_to_try = [',', ';', '\t']
    for encoding in encodings_to_try:
        for sep in delimiters_to_try:
            try:
                game_details_df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                print(f"Successfully loaded game_details_with_image_links as CSV with encoding='{encoding}' and sep='{sep}'.")
                break
            except Exception as csv_err:
                continue
        if game_details_df is not None:
            break
    
    if game_details_df is None:
        print(f"Error: Could not load '{file_path}' with common encodings and delimiters.")


if game_details_df is not None:
    if 'Name' in game_details_df.columns:
        game_details_df['original_title'] = game_details_df['Name']
        game_details_df.rename(columns={'Name': 'game_title_for_normalization'}, inplace=True)
    elif 'game_title' in game_details_df.columns:
        game_details_df['original_title'] = game_details_df['game_title']
        game_details_df.rename(columns={'game_title': 'game_title_for_normalization'}, inplace=True)
    else:
        print("Warning: No 'Name' or 'game_title' column found in game_details_df. Using first column as original_title.")
        game_details_df['original_title'] = game_details_df.iloc[:, 0]
        game_details_df['game_title_for_normalization'] = game_details_df.iloc[:, 0]
        
    game_details_df['game_title_normalized'] = game_details_df['game_title_for_normalization'].apply(normalize_game_title)
    game_details_df.dropna(subset=['game_title_normalized'], inplace=True)
    game_details_df.set_index('game_title_normalized', inplace=True)
    print("Game details loaded and normalized successfully.")
    print("\nColumns in game_details_df after loading and initial processing:")
    print(game_details_df.columns) # Kiểm tra lại tên cột Description
else:
    print("Game details DataFrame is None. Some functionalities might be limited.")


# Hàm để lấy Tên gốc, Mô tả, và URL ảnh cho một game đã chuẩn hóa
def get_game_info(game_name_normalized, description_word_limit=100):
    original_game_title = game_name_normalized
    truncated_description = "No description available for this game."
    full_description = "No description available for this game."
    image_url = None

    if game_details_df is not None and game_name_normalized in game_details_df.index:
        game_info_entry = game_details_df.loc[game_name_normalized]
        
        if isinstance(game_info_entry, pd.DataFrame):
            game_info_entry = game_info_entry.iloc[0]

        if 'original_title' in game_info_entry and pd.notna(game_info_entry['original_title']):
            original_game_title = str(game_info_entry['original_title'])

        # Cải thiện việc lấy mô tả và áp dụng cắt ngắn
        description_text = "No description available for this game."
        # Ưu tiên các tên cột phổ biến cho mô tả
        if 'Description' in game_info_entry and pd.notna(game_info_entry['Description']):
            description_text = str(game_info_entry['Description'])
        elif 'Short_Description' in game_info_entry and pd.notna(game_info_entry['Short_Description']):
            description_text = str(game_info_entry['Short_Description'])
        elif 'long_description' in game_info_entry and pd.notna(game_info_entry['long_description']):
            description_text = str(game_info_entry['long_description'])
        # Thêm các tên cột khác nếu bạn biết tên cột mô tả trong file của mình
        # Ví dụ: elif 'Review Text' in game_info_entry and pd.notna(game_info_entry['Review Text']):
        #            description_text = str(game_info_entry['Review Text'])
            
        truncated_description, full_description = truncate_text(description_text, description_word_limit)
        
        if 'image_link' in game_info_entry and pd.notna(game_info_entry['image_link']):
            image_url = str(game_info_entry['image_link'])
        elif 'AppID' in game_info_entry and pd.notna(game_info_entry['AppID']):
            app_id = int(game_info_entry['AppID'])
            image_url = f"{STEAM_IMAGE_BASE_URL}{app_id}{STEAM_IMAGE_SUFFIX}"
            
    # Fallback cho original_game_title nếu không tìm thấy trong game_details_df
    temp_original_titles = play_df.drop_duplicates(subset=['item_normalized']).set_index('item_normalized')['item'].to_dict()
    original_game_title = original_game_title if original_game_title != game_name_normalized else temp_original_titles.get(game_name_normalized, game_name_normalized)
    
    return original_game_title, truncated_description, full_description, image_url
# Hàm dự đoán rating và áp dụng scaling (không đổi về logic)
def predict_rating(user_id, item_normalized, k=25):
    if user_id not in user_item_matrix.index or item_normalized not in user_item_matrix.columns:
        return scale_rating(play_df['rating'].mean())

    user_sims = user_sim_df[user_id].drop(user_id)
    users_who_rated_item = user_item_matrix[user_item_matrix[item_normalized] > 0].index

    neighbor_sims = user_sims[user_sims.index.isin(users_who_rated_item)]

    top_k_neighbors = neighbor_sims.nlargest(k)

    if top_k_neighbors.empty:
        return scale_rating(play_df['rating'].mean())

    numerator = 0
    denominator = 0
    for neighbor_id, similarity in top_k_neighbors.items():
        neighbor_rating = user_item_matrix.loc[neighbor_id, item_normalized]
        numerator += similarity * neighbor_rating
        denominator += similarity

    if denominator == 0:
        return scale_rating(play_df['rating'].mean())

    raw_predicted_rating = numerator / denominator
    return scale_rating(raw_predicted_rating)

# CÁC HÀM TRẢ VỀ DANH SÁCH GAME SẼ TRẢ VỀ (normalized_title, scaled_rating)
# Sau đó, Flask route sẽ dùng get_game_info để lấy tên gốc và các chi tiết khác.

def get_top_rated_games_with_ratings(num_games=10):
    """Lấy N game có rating trung bình cao nhất, kèm theo rating đã scale."""
    top_games_raw = game_average_ratings_normalized.nlargest(num_games)
    # Trả về (normalized_title, scaled_rating)
    return [(normalized_title, scale_rating(raw_rating)) for normalized_title, raw_rating in top_games_raw.items()]

def get_random_top_rated_games_with_ratings(total_top_games=100, num_random_games=10):
    """Lấy ngẫu nhiên N game từ M game có rating trung bình cao nhất, kèm rating đã scale."""
    top_games_all = get_top_rated_games_with_ratings(total_top_games) # Trả về (normalized_title, scaled_rating)
    if len(top_games_all) < num_random_games:
        return top_games_all
    return random.sample(top_games_all, num_random_games)

def get_game_average_scaled_rating(game_name_normalized):
    """Trả về rating trung bình đã scale của một game (dựa trên tên đã chuẩn hóa)."""
    if game_name_normalized in game_average_ratings_normalized:
        return scale_rating(game_average_ratings_normalized[game_name_normalized])
    return None

def recommend_games_for_user(user_id, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        top_games_raw = game_average_ratings_normalized.nlargest(num_recommendations)
        return [(normalized_title, scale_rating(raw_rating)) for normalized_title, raw_rating in top_games_raw.items()]

    user_rated_games = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()
    possible_recommendations = []

    for item_normalized in all_games_normalized: # item_normalized là tên đã chuẩn hóa
        if item_normalized not in user_rated_games:
            predicted_rating = predict_rating(user_id, item_normalized)
            if predicted_rating > 1:
                possible_recommendations.append((item_normalized, predicted_rating))

    possible_recommendations.sort(key=lambda x: x[1], reverse=True)

    top_potential_recommendations = possible_recommendations[:50] 

    if len(top_potential_recommendations) < num_recommendations:
        final_recommendations = top_potential_recommendations
    else:
        final_recommendations = random.sample(top_potential_recommendations, num_recommendations)

    return final_recommendations