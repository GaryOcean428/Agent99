from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
from memory_manager import memory_manager
from advanced_router import advanced_router
from chat99 import generate_response

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY"
)  # generate key by using import secrets, preint(secrets.token_hex(16)),

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    if "conversation" not in session:
        session["conversation"] = []
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    if "conversation" not in session:
        session["conversation"] = []

    session["conversation"].append({"role": "user", "content": user_input})
    response = generate_response(user_input, session["conversation"])
    session["conversation"].append({"role": "assistant", "content": response})

    # Limit conversation history to last 10 messages
    session["conversation"] = session["conversation"][-10:]
    session.modified = True

    return jsonify({"response": response})


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process the file and add to memory
        with open(file_path, "r") as f:
            content = f.read()
        memory_manager.add_to_long_term_memory(content)

        return jsonify({"message": "File uploaded and processed successfully"})
    return jsonify({"error": "Invalid file type"})


if __name__ == "__main__":
    app.run(debug=True)
