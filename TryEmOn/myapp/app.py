from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Directory where uploaded images will be stored
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variables to store paths of uploaded images
option1_filename = None
option2_filename = None

@app.route('/')
def index():
    # Pass URLs of uploaded images to the template if they exist
    option1_url = url_for('static', filename=f'uploads/{option1_filename}') if option1_filename else None
    option2_url = url_for('static', filename=f'uploads/{option2_filename}') if option2_filename else None
    return render_template('new_index.html', option1_url=option1_url, option2_url=option2_url)

@app.route('/upload_option1', methods=['POST'])
def upload_option1():
    global option1_filename
    file = request.files.get('option1_image')
    if file and file.filename != '':
        option1_filename = secure_filename("option1_" + file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], option1_filename))
    return redirect(url_for('index'))

@app.route('/upload_option2', methods=['POST'])
def upload_option2():
    global option2_filename
    file = request.files.get('option2_image')
    if file and file.filename != '':
        option2_filename = secure_filename("option2_" + file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], option2_filename))
    return redirect(url_for('index'))

@app.route('/select', methods=['POST'])
def select():
    selected_option = request.form.get('selected_option')
    if selected_option == 'option1':
        selected_image = option1_filename
    elif selected_option == 'option2':
        selected_image = option2_filename
    else:
        selected_image = None

    return f"You selected: {selected_image}"

if __name__ == '__main__':
    app.run(debug=True)
