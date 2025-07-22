# Cài đặt scikit-surprise nếu chưa được cài đặt
# Nếu bạn đang chạy trong môi trường Jupyter, có thể dùng %pip.
# Nếu chạy từ script Python thông thường (.py), hãy bỏ % và chạy 'pip install scikit-surprise' trên terminal.
import pandas as pd
import numpy as np
import math

# Nhập các module từ thư viện Surprise
from surprise import KNNBasic
from surprise import Dataset, Reader
# Đổi tên train_test_split từ surprise để tránh xung đột với sklearn
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

# Nhập các module từ sklearn cho phần triển khai thủ công
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# --- 1. Tìm, Chọn Dữ liệu và Xử lý Dữ liệu ---
print("--- 1. Tải và Xử lý Dữ liệu (steam-200k.csv) ---")

# Tải tập dữ liệu steam-200k.csv
# Tập dữ liệu không có header, nên cần chỉ định tên cột
df = pd.read_csv('steam-200k.csv', header=None)
df.columns = ['user_id', 'game_title', 'behavior_name', 'value', 'unused']
print("5 dòng đầu của steam-200k.csv gốc:")
print(df.head())
print("\n")

# Bỏ cột 'unused'
df = df.drop('unused', axis=1)

# Lọc các hàng chỉ có 'play' behavior
play_df = df[df['behavior_name'] == 'play'].copy() # Sử dụng .copy() để tránh cảnh báo SettingWithCopyWarning
print("5 dòng đầu của play_df (chỉ behavior 'play'):")
print(play_df.head())
print("\n")

# Bỏ cột 'behavior_name' vì nó luôn là 'play'
play_df = play_df.drop('behavior_name', axis=1)

# Đổi tên cột thành 'user', 'item', 'rating' theo chuẩn của các hệ thống đề xuất
play_df.rename(columns={'user_id': 'user', 'game_title': 'item', 'value': 'rating'}, inplace=True)

# Áp dụng biến đổi log1p cho cột 'rating' (thời gian chơi) để chuẩn hóa phân phối
play_df['rating'] = np.log1p(play_df['rating'])
print("5 dòng đầu của play_df sau biến đổi log1p và đổi tên cột:")
print(play_df.head())
print(f"Khoảng giá trị của rating sau log1p: Nhỏ nhất={play_df['rating'].min():.4f}, Lớn nhất={play_df['rating'].max():.4f}\n")


# --- 2. Triển khai: Sử dụng thư viện Surprise ---
print("--- 2. Triển khai sử dụng thư viện Surprise ---")

# Tạo đối tượng Reader cho Surprise Dataset
# rating_scale phải khớp với min/max của cột 'rating' đã biến đổi log1p
min_rating_surprise = play_df['rating'].min()
max_rating_surprise = play_df['rating'].max()
reader = Reader(rating_scale=(min_rating_surprise, max_rating_surprise))

# Tải dữ liệu từ play_df vào một đối tượng Surprise Dataset
data_surprise = Dataset.load_from_df(play_df[['user', 'item', 'rating']], reader)

# Chia tập dữ liệu thành trainset và testset
# SỬ DỤNG surprise_train_test_split ĐÃ ĐỔI TÊN
trainset_surprise, testset_surprise = surprise_train_test_split(data_surprise, test_size=.25, random_state=42) # Sử dụng 25% làm tập kiểm tra

print(f"Tổng số {trainset_surprise.n_users} người dùng và {trainset_surprise.n_items} mục trong tập huấn luyện của Surprise\n")

# TASK: Thực hiện lọc cộng tác dựa trên KNN trên ma trận tương tác người dùng-mục
# TODO: Huấn luyện mô hình lọc cộng tác dựa trên KNN bằng trainset và đánh giá kết quả bằng testset:

# - Định nghĩa một mô hình KNNBasic()
# Thử các kết hợp hyperparameter khác nhau để xem cái nào có hiệu suất tốt nhất
# Ví dụ: user_based = True (Lọc cộng tác dựa trên người dùng)
print("Huấn luyện mô hình KNNBasic dựa trên người dùng với Surprise...")
algo_user_based_surprise = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
# - Huấn luyện mô hình KNNBasic trên trainset
algo_user_based_surprise.fit(trainset_surprise)
# - Dự đoán rating cho testset
predictions_user_based_surprise = algo_user_based_surprise.test(testset_surprise)
# - Sau đó tính RMSE
print("RMSE của KNNBasic dựa trên người dùng (Surprise):")
accuracy.rmse(predictions_user_based_surprise, verbose=True)
print("\n")

# Ví dụ: user_based = False (Lọc cộng tác dựa trên mục)
print("Huấn luyện mô hình KNNBasic dựa trên mục với Surprise...")
algo_item_based_surprise = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
algo_item_based_surprise.fit(trainset_surprise)
predictions_item_based_surprise = algo_item_based_surprise.test(testset_surprise)
print("RMSE của KNNBasic dựa trên mục (Surprise):")
accuracy.rmse(predictions_item_based_surprise, verbose=True)
print("\n")

# --- 3. Triển khai: Sử dụng numpy, pandas và sklearn ---
print("--- 3. Triển khai sử dụng numpy, pandas và sklearn (KNN thủ công) ---")

# Chia play_df thành các tập huấn luyện và kiểm tra
# SỬ DỤNG sklearn_train_test_split ĐÃ ĐỔI TÊN
train_df_manual, test_df_manual = sklearn_train_test_split(play_df, test_size=0.25, random_state=42) # Sử dụng 25% làm tập kiểm tra

# - Tính toán độ tương đồng giữa hai người dùng bằng lịch sử rating của họ
# - Xây dựng ma trận độ tương đồng cho mỗi cặp người dùng với tập dữ liệu huấn luyện
train_user_item_matrix = train_df_manual.pivot_table(index='user', columns='item', values='rating', aggfunc='sum').fillna(0)
print("5 dòng đầu của Ma trận Người dùng-Mục huấn luyện (KNN thủ công):")
print(train_user_item_matrix.head())
print("\n")

user_similarity_matrix = cosine_similarity(train_user_item_matrix)
user_sim_df_manual = pd.DataFrame(user_similarity_matrix,
                                  index=train_user_item_matrix.index,
                                  columns=train_user_item_matrix.index)
print("5 dòng đầu của DataFrame Độ tương đồng Người dùng (KNN thủ công):")
print(user_sim_df_manual.head())
print("\n")

def predict_rating_manual(user_id, item_id, train_matrix, sim_df, k=25):
    """
    Dự đoán rating cho một người dùng và mục cụ thể bằng lọc cộng tác dựa trên người dùng thủ công.
    """
    if user_id not in train_matrix.index or item_id not in train_matrix.columns:
        # Nếu người dùng hoặc mục không có trong ma trận huấn luyện, trả về rating trung bình từ train_df_manual
        return train_df_manual['rating'].mean()
    
    # - Đối với mỗi người dùng, tìm k hàng xóm gần nhất trong ma trận độ tương đồng
    user_sims = sim_df[user_id].drop(user_id)
    
    # Xác định những người dùng đã đánh giá mục cụ thể
    users_who_rated_item = train_matrix[train_matrix[item_id] > 0].index
    
    # Lọc những người dùng tương tự chỉ còn những người đã đánh giá mục
    neighbor_sims = user_sims[user_sims.index.isin(users_who_rated_item)]
    
    # Chọn k hàng xóm gần nhất
    top_k_neighbors = neighbor_sims.nlargest(k)
    
    if top_k_neighbors.empty:
        # Nếu không có hàng xóm phù hợp, trả về rating trung bình từ train_df_manual
        return train_df_manual['rating'].mean()
        
    numerator = 0
    denominator = 0
    # - Đối với mỗi rating trong tập dữ liệu kiểm tra, ước tính rating của nó bằng các phương trình lọc cộng tác KNN đã hiển thị trước đó
    for neighbor_id, similarity in top_k_neighbors.items():
        neighbor_rating = train_matrix.loc[neighbor_id, item_id]
        numerator += similarity * neighbor_rating
        denominator += similarity
        
    if denominator == 0:
        # Tránh chia cho 0, trả về rating trung bình từ train_df_manual
        return train_df_manual['rating'].mean()
        
    return numerator / denominator

# Thực hiện dự đoán cho tập kiểm tra
test_df_manual['predicted_rating'] = test_df_manual.apply(
    lambda row: predict_rating_manual(row['user'], row['item'], train_user_item_matrix, user_sim_df_manual, k=25),
    axis=1
)

# - Tính RMSE cho toàn bộ tập dữ liệu kiểm tra
rmse_manual = np.sqrt(mean_squared_error(test_df_manual['rating'], test_df_manual['predicted_rating']))
print(f"RMSE của KNN dựa trên người dùng thủ công: {rmse_manual:.4f}")
print("\n")

print("--- Kết thúc Triển khai ---")