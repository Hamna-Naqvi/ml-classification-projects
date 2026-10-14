"""Flask web app. Run: python app_flask.py  (opens on http://127.0.0.1:5000)"""
from flask import Flask, render_template_string, request

from common import DEFAULTS, NEG, POS, TITLE, load, predict_proba

model, features, acc = load()
app = Flask(__name__)

PAGE = """
<!doctype html>
<title>{{ title }}</title>
<style>
  body{font-family:Arial;max-width:520px;margin:30px auto;padding:0 12px}
  label{display:block;margin-top:10px;font-weight:bold}
  input{width:100%;padding:7px;box-sizing:border-box}
  button{margin-top:16px;padding:10px 18px;cursor:pointer}
  .res{margin-top:20px;padding:14px;background:#eef;border-radius:6px}
</style>
<h2>{{ title }}</h2>
<p>Logistic Regression &middot; test accuracy {{ acc }}%</p>
<form method="post">
  {% for f in features %}
    <label>{{ f }}</label>
    <input type="number" step="any" name="{{ f }}" value="{{ vals[f] }}" required>
  {% endfor %}
  <button type="submit">Predict</button>
</form>
{% if result %}
  <div class="res"><b>{{ result }}</b><br>Probability: {{ prob }}%</div>
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    vals = {f: DEFAULTS.get(f, 0) for f in features}
    result = prob = None
    if request.method == "POST":
        vals = {f: float(request.form[f]) for f in features}
        p = predict_proba(model, features, [vals[f] for f in features])
        result = POS if p >= 0.5 else NEG
        prob = round(max(p, 1 - p) * 100, 2)
    return render_template_string(
        PAGE, title=TITLE, features=features, vals=vals,
        result=result, prob=prob, acc=round(acc * 100, 1),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
