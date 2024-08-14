from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = /path/to/the/uploads
app.config[UPLOAD_FOLDER] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_EXTENSIONS = {txt, pdf, png, jpg, jpeg, gif}

def allowed_file(filename):
    return . in filename and filename.rsplit(., 1).lower() in ALLOWED_EXTENSIONS

@app.route(/upload, methods=[POST])
def upload_file():
    if request.method == POST:
        # Check if the post request has the file part
        if file not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files[file]
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == :
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config[UPLOAD_FOLDER], filename))
            return jsonify({"message": "File uploaded successfully"}), 200
    return jsonify({"error": "Invalid request"}), 400
