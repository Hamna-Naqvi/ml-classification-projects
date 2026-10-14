"""Gradio web app. Run: python app_gradio.py  (opens on http://127.0.0.1:7860)"""
import gradio as gr

from common import DEFAULTS, NEG, NOTE, POS, TITLE, load, predict_proba

model, features, acc = load()


def predict(*values):
    p = predict_proba(model, features, values)
    return {POS: p, NEG: 1 - p}


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=f, value=DEFAULTS.get(f, 0)) for f in features],
    outputs=gr.Label(num_top_classes=2),
    title=TITLE,
    description=f"Logistic Regression | test accuracy: {acc * 100:.1f}%. {NOTE}",
)

if __name__ == "__main__":
    demo.launch()
