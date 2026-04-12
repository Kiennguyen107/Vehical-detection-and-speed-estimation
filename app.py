from flask import Flask, render_template, request, send_file
from database import get_conn
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results      = []
    dist_results = []
    plate        = ''

    if request.method == 'POST':
        plate = request.form.get('plate', '').strip().upper()
        if plate:
            conn = get_conn()
            cur  = conn.cursor()

            # Query bảng vi phạm tốc độ
            cur.execute("""
                SELECT time, date, speed, lane, plate, color,
                       status, frame_path, dist_violation, over_by
                FROM speed_violations
                WHERE REPLACE(REPLACE(UPPER(plate), '-', ''), '.', '') LIKE %s
                ORDER BY date DESC, time DESC
            """, (f'%{plate}%',))
            results = cur.fetchall()

            # Query bảng vi phạm khoảng cách
            cur.execute("""
                SELECT time, date, distance, safe_dist, speed, lane, plate
                FROM distance_violations
                WHERE REPLACE(REPLACE(UPPER(plate), '-', ''), '.', '') LIKE %s
                ORDER BY date DESC, time DESC
            """, (f'%{plate}%',))
            dist_results = cur.fetchall()

            cur.close()
            conn.close()

    return render_template('index.html',
                           results=results,
                           dist_results=dist_results,
                           plate=plate)

@app.route('/image')
def serve_image():
    path = request.args.get('path', '')
    if path and os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return '', 404

if __name__ == '__main__':
    app.run(debug=True)