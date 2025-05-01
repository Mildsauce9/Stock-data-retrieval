from flask import Flask, render_template, request
from model import process_user_query

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        q = request.form.get("query", "")
        answer = process_user_query(q)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)