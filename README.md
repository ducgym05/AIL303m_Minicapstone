# Hệ thống Đề xuất Game Steam (Steam Game Recommender System)

## Giới thiệu

Đây là một dự án mini-capstone xây dựng hệ thống đề xuất game sử dụng thuật toán lọc cộng tác (Collaborative Filtering). Hệ thống được phát triển bằng Python với Flask làm framework web và sử dụng thư viện `Surprise` cùng với triển khai thủ công `NumPy`/`Pandas`/`scikit-learn` để so sánh hiệu suất.

Mục tiêu của hệ thống là đề xuất các game mà người dùng có thể thích, dựa trên hành vi chơi game của họ.

## Các tính năng chính

* **Đề xuất game dựa trên lọc cộng tác (Collaborative Filtering):**
    * [cite_start]Sử dụng thuật toán K-Nearest Neighbors (KNN) để tìm người dùng (hoặc game) tương tự[cite: 1].
    * Hỗ trợ cả hai phương pháp:
        * [cite_start]**User-based Collaborative Filtering:** Dựa trên độ tương đồng giữa các người dùng[cite: 3].
        * [cite_start]**Item-based Collaborative Filtering:** Dựa trên độ tương đồng giữa các mục (game)[cite: 4].
* **Xử lý và chuẩn hóa dữ liệu:**
    * Tải dữ liệu hành vi chơi game từ `steam-200k.csv`.
    * Lọc các bản ghi `play` và biến đổi `playtime` thành `rating` bằng `np.log1p` để chuẩn hóa phân phối.
    * Loại bỏ các giá trị outliers (giờ chơi gốc trên 3000 giờ) để cải thiện chất lượng dữ liệu.
    * Áp dụng các kỹ thuật Scaling (StandardScaler, RobustScaler, MinMaxScaler) cho dữ liệu rating để tối ưu hóa hiệu suất mô hình.
    * Chuẩn hóa tên game thành dạng chuỗi thống nhất để liên kết dữ liệu chính xác giữa `steam-200k.csv` và `game_details_with_image_links.csv`.
* **Giao diện Web Local (Flask):**
    * **Trang chủ:** Hiển thị 10 game phổ biến được chọn ngẫu nhiên với tên gốc, rating và ảnh nền của game đó.
    * **Tìm kiếm game:** Cho phép người dùng tìm kiếm game theo tên (tìm kiếm một phần tên), hiển thị kết quả với tên gốc và rating.
    * **Trang chi tiết game:** Khi nhấp vào một game:
        * Hiển thị tên gốc của game, rating trung bình, mô tả và ảnh đại diện.
        * Mô tả game được cắt ngắn 100 từ và có nút "Read more" để xem toàn bộ.
        * Nền của phần thông tin game được thay bằng ảnh của game đó.
        * **Phần đề xuất:** Gợi ý 5 game khác dựa trên hành vi người dùng, mỗi game đề xuất cũng có tên gốc, rating và ảnh nền riêng.
    * **Top 10 game tổng thể:** Liệt kê 10 game có rating cao nhất trong toàn bộ tập dữ liệu.
    * Giao diện được thiết kế hiện đại, tối giản và sang trọng.
* **Đánh giá mô hình:** Tính toán RMSE (Root Mean Square Error) để đánh giá hiệu suất dự đoán của cả hai triển khai (Surprise và thủ công) cho từng phương pháp scaling.

## Cấu trúc dự án

Here's a README.md file for your Steam Game Recommender project, incorporating the details and functionalities you've implemented:

Markdown

# Hệ thống Đề xuất Game Steam (Steam Game Recommender System)

## Giới thiệu

Đây là một dự án mini-capstone xây dựng hệ thống đề xuất game sử dụng thuật toán lọc cộng tác (Collaborative Filtering). Hệ thống được phát triển bằng Python với Flask làm framework web và sử dụng thư viện `Surprise` cùng với triển khai thủ công `NumPy`/`Pandas`/`scikit-learn` để so sánh hiệu suất.

Mục tiêu của hệ thống là đề xuất các game mà người dùng có thể thích, dựa trên hành vi chơi game của họ.

## Các tính năng chính

* **Đề xuất game dựa trên lọc cộng tác (Collaborative Filtering):**
    * [cite_start]Sử dụng thuật toán K-Nearest Neighbors (KNN) để tìm người dùng (hoặc game) tương tự[cite: 1].
    * Hỗ trợ cả hai phương pháp:
        * [cite_start]**User-based Collaborative Filtering:** Dựa trên độ tương đồng giữa các người dùng[cite: 3].
        * [cite_start]**Item-based Collaborative Filtering:** Dựa trên độ tương đồng giữa các mục (game)[cite: 4].
* **Xử lý và chuẩn hóa dữ liệu:**
    * Tải dữ liệu hành vi chơi game từ `steam-200k.csv`.
    * Lọc các bản ghi `play` và biến đổi `playtime` thành `rating` bằng `np.log1p` để chuẩn hóa phân phối.
    * Loại bỏ các giá trị outliers (giờ chơi gốc trên 3000 giờ) để cải thiện chất lượng dữ liệu.
    * Áp dụng các kỹ thuật Scaling (StandardScaler, RobustScaler, MinMaxScaler) cho dữ liệu rating để tối ưu hóa hiệu suất mô hình.
    * Chuẩn hóa tên game thành dạng chuỗi thống nhất để liên kết dữ liệu chính xác giữa `steam-200k.csv` và `game_details_with_image_links.csv`.
* **Giao diện Web Local (Flask):**
    * **Trang chủ:** Hiển thị 10 game phổ biến được chọn ngẫu nhiên với tên gốc, rating và ảnh nền của game đó.
    * **Tìm kiếm game:** Cho phép người dùng tìm kiếm game theo tên (tìm kiếm một phần tên), hiển thị kết quả với tên gốc và rating.
    * **Trang chi tiết game:** Khi nhấp vào một game:
        * Hiển thị tên gốc của game, rating trung bình, mô tả và ảnh đại diện.
        * Mô tả game được cắt ngắn 100 từ và có nút "Read more" để xem toàn bộ.
        * Nền của phần thông tin game được thay bằng ảnh của game đó.
        * **Phần đề xuất:** Gợi ý 5 game khác dựa trên hành vi người dùng, mỗi game đề xuất cũng có tên gốc, rating và ảnh nền riêng.
    * **Top 10 game tổng thể:** Liệt kê 10 game có rating cao nhất trong toàn bộ tập dữ liệu.
    * Giao diện được thiết kế hiện đại, tối giản và sang trọng.
* **Đánh giá mô hình:** Tính toán RMSE (Root Mean Square Error) để đánh giá hiệu suất dự đoán của cả hai triển khai (Surprise và thủ công) cho từng phương pháp scaling.

## Cấu trúc dự án

.
├── .gitignore
├── app.py
├── cong_en.py
├── data_review.ipynb
├── game_details_with_image_links.csv
├── README.md
├── recommend_core.py
├── steam-200k.csv
├── static/
│   └── style.css
└── templates/
    ├── game_detail.html
    └── index.html

## Yêu cầu hệ thống

* Python 3.7+
* Các thư viện Python (được liệt kê trong `requirements.txt`):
    * `pandas`
    * `numpy`
    * `scikit-learn`
    * `flask`
    * `scikit-surprise`
    * `matplotlib`
    * `seaborn`
    * `openpyxl` (nếu `game_details_with_image_links.csv` thực chất là file `.xlsx`)
    * `gunicorn` (cho triển khai production)

## Cài đặt

1.  **Clone repository (nếu có) hoặc tải xuống các file dự án.**
2.  **Cài đặt các thư viện Python:**
    ```bash
    pip install -r requirements.txt
    ```
    (Nếu bạn chưa có `requirements.txt`, hãy tạo nó bằng `pip freeze > requirements.txt` sau khi cài đặt thủ công tất cả các thư viện cần thiết: `pip install pandas numpy scikit-learn flask scikit-surprise matplotlib seaborn openpyxl gunicorn`)
3.  **Đặt các file dữ liệu:** Đảm bảo `steam-200k.csv` và `game_details_with_image_links.csv` nằm cùng thư mục gốc của dự án với `app.py` và `recommend_core.py`.
    * **LƯU Ý:** File `game_details_with_image_links.csv` của bạn cần được định dạng chính xác với các cột:
        * `Name` (hoặc tên cột chứa tên game gốc)
        * `Description` (tên cột chứa mô tả game)
        * `image_link` (URL trực tiếp đến ảnh, ưu tiên)
        * `AppID` (ID của game, dùng làm fallback cho link ảnh Steam)

## Cách chạy ứng dụng

1.  **Mở Terminal/Command Prompt** và điều hướng đến thư mục gốc của dự án.
2.  **Chạy ứng dụng Flask:**
    ```bash
    python app.py
    ```
3.  **Tru cập trên trình duyệt:** Mở trình duyệt web của bạn và truy cập địa chỉ: `http://127.0.0.1:5000/` (hoặc địa chỉ được hiển thị trong terminal).

## Triển khai (Deployment)

Để ứng dụng có thể được truy cập công khai qua một đường link, bạn cần triển khai nó lên một máy chủ web. Các tùy chọn phổ biến bao gồm Heroku, PythonAnywhere, Render.com hoặc Glitch.com.

* Bạn có thể cần tạo một file `Procfile` (không có phần mở rộng) trong thư mục gốc với nội dung:
    ```
    web: gunicorn app:app
    ```
    (Đối với môi trường Linux/WSL. Nếu dùng Windows, bạn có thể cần `waitress` cho local testing hoặc chọn dịch vụ hosting hỗ trợ Windows.)

## Đánh giá hiệu suất RMSE

Để chạy phần đánh giá RMSE của các mô hình với các scaler khác nhau:

1.  Đảm bảo đã cài đặt tất cả các thư viện.
2.  Mở một môi trường phát triển Python (ví dụ: Jupyter Notebook, VS Code với Python Extension) và mở file `recommend_core.py`.
3.  Bạn có thể chạy các phần code trong `recommend_core.py` để xem các bước tiền xử lý dữ liệu và kết quả RMSE cho từng loại scaler và triển khai mô hình.

## Tín dụng

* Dữ liệu: `steam-200k.csv`
* [cite_start]Thuật toán lọc cộng tác: K-Nearest Neighbors (KNN) [cite: 1, 18]
* Thư viện: `pandas`, `numpy`, `scikit-learn`, `flask`, `scikit-surprise`
* [cite_start]Tài liệu tham khảo về Collaborative Filtering và KNN: [cite: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

---
**Lưu ý:** Nếu bạn chưa có file `requirements.txt`, hãy tạo nó bằng cách chạy `pip freeze > requirements.txt` trong terminal tại thư mục gốc dự án sau khi bạn đã cài đặt tất cả các thư viện cần thiết.