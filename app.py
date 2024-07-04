from flask import Flask, redirect, url_for, request, session, render_template
import requests
from PIL import Image
import numpy as np
from tensorflow.lite.python import interpreter
import pandas as pd

app = Flask(__name__)
app.secret_key = '\t\x0cp\xbb\xb8V\xbd+J\xa5a\x97_6\xb7l)\x98@\x1a\x9d0s\xee'

# Google OAuth configuration
CLIENT_ID = 'xxx'
CLIENT_SECRET = 'xxx'

REDIRECT_URI = 'https://bornobyte.sarwaruddin.net/google/callback'
AUTH_URI = 'https://accounts.google.com/o/oauth2/auth'
TOKEN_URI = 'https://oauth2.googleapis.com/token'
USER_INFO_URI = 'https://www.googleapis.com/oauth2/v3/userinfo'

# Load TensorFlow Lite model
interpreter = interpreter.Interpreter(model_path='bornobyte.tflite')
interpreter.allocate_tensors()

# Load metadata
csv_file_path = 'metaData_img.csv'
df = pd.read_csv(csv_file_path)
folder_to_char = dict(zip(df['Folder Name'], df['Char Name']))

@app.route('/')
def index():
    username = session.get('username')
    return render_template('index.html', username=username)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    if 'access_token' in session:
        return redirect(url_for('index'))

    auth_url = f"{AUTH_URI}?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&response_type=code&scope=email%20profile"
    return redirect(auth_url)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('logged_out'))

@app.route('/logged_out')
def logged_out():
    return render_template('logged_out.html')

@app.route('/google/callback')
def callback():
    auth_code = request.args.get('code')
    token_url = f"{TOKEN_URI}"
    token_data = {
        'code': auth_code,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code'
    }
    token_response = requests.post(token_url, data=token_data)
    access_token = token_response.json().get('access_token')
    if not access_token:
        return 'Failed to get access token', 400

    session['access_token'] = access_token
    user_info_response = requests.get(USER_INFO_URI, headers={'Authorization': f'Bearer {access_token}'})
    user_info = user_info_response.json()
    if 'error' in user_info:
        return 'Failed to fetch user info', 400

    session['username'] = user_info.get('name')
    session['email'] = user_info.get('email')
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream)
            processed_img = preprocess_image(img)
            prediction = run_inference(processed_img)
            predicted_character = folder_to_char.get(prediction, 'Unknown')
            return render_template('index.html', prediction=predicted_character)

    return redirect(url_for('index'))

def preprocess_image(image):
    resized_image = image.resize((28, 28)).convert('L')
    normalized_image = np.array(resized_image) / 255.0
    processed_img = np.expand_dims(normalized_image, axis=0)
    processed_img = processed_img.astype(np.float32)
    processed_img = np.expand_dims(processed_img, axis=-1)
    return processed_img


def run_inference(input_data):
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)

