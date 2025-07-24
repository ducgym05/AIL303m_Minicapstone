# app.py

from flask import Flask, render_template, request, url_for
import recommend_core as rec_sys
import os

app = Flask(__name__)
WORD_LIMIT = 100

def format_games_for_display(list_of_normalized_titles):
    display_list = []
    for norm_title in list_of_normalized_titles:
        orig_title, _, image_path = rec_sys.get_game_info(norm_title)
        display_list.append({
            'original_title': orig_title,
            'rating': rec_sys.get_game_rating(norm_title),
            'normalized_title': norm_title,
            'image_path': image_path or url_for('static', filename='placeholder.jpg')
        })
    return display_list

@app.route('/')
def index():
    random_games_normalized = rec_sys.get_random_games_from_top(num_games=12)
    games_for_display = format_games_for_display(random_games_normalized)
    top_200_random_normalized = rec_sys.get_random_games_from_top(num_games=20)
    top_200_for_display = format_games_for_display(top_200_random_normalized)
    return render_template('index.html',
                           games=games_for_display,
                           top_200_games=top_200_for_display,
                           show_sidebar=True, # BẬT sidebar ở trang chủ
                           show_random_games=True,
                           page_title="Discover Games")

@app.route('/top_overall_games')
def top_overall_games():
    top_games_normalized = rec_sys.get_top_rated_games(num_games=10)
    games_for_display = format_games_for_display(top_games_normalized)
    return render_template('index.html',
                           games=games_for_display,
                           show_sidebar=False, # TẮT sidebar ở trang top 10
                           show_top_overall=True,
                           page_title="Top 10 Overall Games")

@app.route('/search')
def search():
    query = request.args.get('query', '').strip()
    games_for_display = []
    if query:
        found_games_normalized = rec_sys.search_games(query)
        games_for_display = format_games_for_display(found_games_normalized)
    return render_template('index.html',
                           search_results=games_for_display,
                           show_sidebar=False, # TẮT sidebar ở trang tìm kiếm
                           query=query,
                           show_search_results=True)

# (Phần còn lại của app.py, bao gồm route 'game_detail', được giữ nguyên)
@app.route('/game/<game_name_normalized>')
def game_detail(game_name_normalized):
    original_title, full_description, image_path = rec_sys.get_game_info(game_name_normalized)
    current_game_rating = rec_sys.get_game_rating(game_name_normalized)
    words = full_description.split()
    if len(words) > WORD_LIMIT:
        truncated_description = ' '.join(words[:WORD_LIMIT]) + '...'
    else:
        truncated_description = full_description
    recommended_normalized = rec_sys.get_recommendations(game_name_normalized)
    recommended_normalized = [game for game in recommended_normalized if game != game_name_normalized]
    recommended_games_for_display = format_games_for_display(recommended_normalized)
    return render_template('game_detail.html',
                           game_name=original_title,
                           game_image_path=image_path or url_for('static', filename='placeholder.jpg'),
                           full_description=full_description,
                           truncated_description=truncated_description,
                           current_game_rating=current_game_rating,
                           recommended_games=recommended_games_for_display)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)