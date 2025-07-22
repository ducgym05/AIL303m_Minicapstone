from flask import Flask, render_template, request, redirect, url_for
import your_recommendation_logic as rec_sys
import random
import os

app = Flask(__name__)

# Helper function to get full game info for display in templates
def get_display_games_info(game_list_normalized_ratings):
    display_list = []
    for normalized_title, rating in game_list_normalized_ratings:
        # Gọi get_game_info để lấy tất cả chi tiết
        original_title, truncated_description, full_description, image_path = rec_sys.get_game_info(normalized_title)
        display_list.append({
            'original_title': original_title,
            'rating': rating,
            'normalized_title': normalized_title,
            'image_path': image_path if image_path else url_for('static', filename='placeholder.jpg') # Đảm bảo luôn có ảnh hoặc placeholder
        })
    return display_list

# Trang chủ: Hiển thị 10 game ngẫu nhiên trong 100 game có rating cao nhất
@app.route('/')
def index():
    random_games_normalized_ratings = rec_sys.get_random_top_rated_games_with_ratings(total_top_games=100, num_random_games=10)
    games_for_display = get_display_games_info(random_games_normalized_ratings)
    return render_template('index.html', games=games_for_display, show_random_games=True)

# Trang hiển thị 10 game có rating cao nhất
@app.route('/top_overall_games')
def top_overall_games():
    top_games_normalized_ratings = rec_sys.get_top_rated_games_with_ratings(num_games=10)
    games_for_display = get_display_games_info(top_games_normalized_ratings)
    return render_template('index.html', games=games_for_display, show_top_overall=True)

# Tìm kiếm game
@app.route('/search', methods=['GET'])
def search_game():
    query = request.args.get('query', '').strip()
    normalized_query = rec_sys.normalize_game_title(query)
    
    found_games_normalized_ratings = []
    if normalized_query:
        for game_name_normalized in rec_sys.all_games_normalized:
            if normalized_query in game_name_normalized:
                avg_scaled_rating = rec_sys.get_game_average_scaled_rating(game_name_normalized)
                if avg_scaled_rating is not None:
                    found_games_normalized_ratings.append((game_name_normalized, avg_scaled_rating))
    
    search_results_for_display = get_display_games_info(found_games_normalized_ratings)
    return render_template('index.html', search_results=search_results_for_display, query=query, show_search_results=True)

# Trang chi tiết game và đề xuất
@app.route('/game/<game_name_normalized>')
def game_detail(game_name_normalized):
    if game_name_normalized not in rec_sys.all_games_normalized:
        return "Game not found", 404

    # Lấy tên gốc, mô tả (cả cắt ngắn và đầy đủ) và đường dẫn ảnh
    original_game_title, truncated_description, full_description, game_image_path = rec_sys.get_game_info(game_name_normalized)
    
    # Lấy rating của game hiện tại để hiển thị
    current_game_rating = rec_sys.get_game_average_scaled_rating(game_name_normalized)

    sample_user_id = random.choice(rec_sys.user_item_matrix.index.tolist())

    recommended_games_normalized_ratings = rec_sys.recommend_games_for_user(sample_user_id, num_recommendations=5)
    
    unique_recommended_games_for_display = []
    seen_normalized_titles = set()
    
    for normalized_rec_title, rating in recommended_games_normalized_ratings:
        if normalized_rec_title != game_name_normalized and normalized_rec_title not in seen_normalized_titles:
            # Lấy thông tin đầy đủ cho game được đề xuất
            rec_original_title, rec_truncated_description, rec_full_description, rec_image_path = rec_sys.get_game_info(normalized_rec_title)
            unique_recommended_games_for_display.append({
                'original_title': rec_original_title,
                'rating': rating,
                'normalized_title': normalized_rec_title,
                'image_path': rec_image_path if rec_image_path else url_for('static', filename='placeholder.jpg') # Đảm bảo luôn có ảnh
            })
            seen_normalized_titles.add(normalized_rec_title)
    
    final_recommendations_for_display = unique_recommended_games_for_display[:5]

    if len(final_recommendations_for_display) < 5:
        top_popular_games_normalized_ratings = rec_sys.get_top_rated_games_with_ratings(num_games=10)
        
        for normalized_pop_title, rating_pop in top_popular_games_normalized_ratings:
            if normalized_pop_title != game_name_normalized and normalized_pop_title not in seen_normalized_titles:
                pop_original_title, pop_truncated_description, pop_full_description, pop_image_path = rec_sys.get_game_info(normalized_pop_title)
                final_recommendations_for_display.append({
                    'original_title': pop_original_title,
                    'rating': rating_pop,
                    'normalized_title': normalized_pop_title,
                    'image_path': pop_image_path if pop_image_path else url_for('static', filename='placeholder.jpg') # Đảm bảo luôn có ảnh
                })
                seen_normalized_titles.add(normalized_pop_title)
            if len(final_recommendations_for_display) >= 5:
                break

    return render_template('game_detail.html', 
                           game_name=original_game_title, # Tên gốc để hiển thị
                           game_name_normalized=game_name_normalized, # Tên chuẩn hóa cho url_for
                           truncated_description=truncated_description, # Mô tả cắt ngắn
                           full_description=full_description, # Mô tả đầy đủ
                           game_image_path=game_image_path if game_image_path else url_for('static', filename='placeholder.jpg'), # Đảm bảo ảnh chính cũng có placeholder
                           current_game_rating=current_game_rating,
                           recommended_games=final_recommendations_for_display)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)