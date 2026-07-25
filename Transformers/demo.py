import gradio as gr
from inference import Translator

# Automatically loads latest local weights or pass weights_path="weights/tmodel_24.pt"
translator = Translator(weights_path="weights/tmodel_24.pt")

def translate_ui(text):
    if not text.strip():
        return "Please enter valid English text."
    return translator.translate(text)

demo = gr.Interface(
    fn=translate_ui,
    inputs=gr.Textbox(lines=2, placeholder="Type English text here...", label="English Input"),
    outputs=gr.Textbox(lines=2, label="Italian Translation"),
    title="English-to-Italian Transformer from Scratch",
    description="Custom PyTorch Seq2Seq Transformer architecture (Encoder-Decoder, Multi-Head Attention, Beam Search) built without PyTorch's nn.Transformer.",
    examples=[
        ["Books are full of knowledge."],
        ["The cat sat on the mat."],
        ["Hello, how are you today?"]
    ]
)

if __name__ == "__main__":
    demo.launch()