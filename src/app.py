# app.py
from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
    template_folder=os.path.join(current_dir, 'templates'))

# Configure upload folder
UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(file_path)
            
            # Import prediction function here to avoid circular imports
            from predict import predict_image
            
            # Get prediction
            prediction, confidence = predict_image(file_path)
            
            # Clean up - remove the uploaded file
            os.remove(file_path)
            
            return jsonify({
                'prediction': prediction,
                'confidence': f"{confidence}%"
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")  # For debugging
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)