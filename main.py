import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_PATH = os.path.join(BASE_DIR, 'results.json')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'change-this-secret-for-production'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception:
            return []


def save_result(record):
    results = load_results()
    results.insert(0, record)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


@app.route('/')
def index():
    info = {
        'name': 'Your Name',
        'about': 'This is a simple Flask app to upload images and view results.',
        'web_info': 'Running locally. Edit `main.py` to change displayed info.'
    }
    return render_template('index.html', info=info)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # avoid collisions by prefixing timestamp
            ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
            saved_name = f"{ts}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
            # ensure upload dir exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(save_path)

            record = {
                'filename': saved_name,
                'original_filename': filename,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            save_result(record)
            # After saving, go to results page
            return redirect(url_for('results'))
        else:
            flash('File type not allowed')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/results')
def results():
    results = load_results()
    return render_template('results.html', results=results)


if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Ensure results.json exists
    if not os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f)
    app.run(host='0.0.0.0', port=5000, debug=True)
