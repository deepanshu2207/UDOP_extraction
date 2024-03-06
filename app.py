import gradio as gr
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, text):
    encoding = processor(image, text, return_tensors="pt")
    
    # forward pass
    with torch.no_grad():
     outputs = model(**encoding)
     
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    predicted_answer = model.config.id2label[idx]
   
    return predicted_answer
   
image = gr.Image(type="pil")
question = gr.Textbox(label="Question")
answer = gr.Textbox(label="Predicted answer")
examples = [["cats.jpg", "How many cats are there?"]]

title = "Interactive demo: ViLT"
description = "Gradio Demo for ViLT (Vision and Language Transformer), fine-tuned on VQAv2, a model that can answer questions from images. To use it, simply upload your image and type a question and click 'submit', or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2102.03334' target='_blank'>ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision</a> | <a href='https://github.com/dandelin/ViLT' target='_blank'>Github Repo</a></p>"

interface = gr.Interface(fn=answer_question, 
                         inputs=[image, question], 
                         outputs=answer, 
                         examples=examples, 
                         title=title,
                         description=description,
                         article=article)
interface.launch(debug=True)