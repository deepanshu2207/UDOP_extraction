import gradio as gr
from transformers import UdopProcessor, UdopForConditionalGeneration
import torch

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

repo_id = "microsoft/udop-large"
processor = UdopProcessor.from_pretrained(repo_id)
model = UdopForConditionalGeneration.from_pretrained(repo_id)


def answer_question(img, user_query):
    encoding = processor(images=img, text=user_query, return_tensors="pt")
    outputs = model.generate(**encoding, max_new_tokens=20)
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
   
    return generated_text
   
image = gr.Image(type="pil")
question = gr.Textbox(label="Question")
answer = gr.Textbox(label="Predicted answer")
examples = [["cats.jpg", "How many cats are there?"]]

title = "Interactive demo: UDOP"
description = "Gradio Demo for UDOP, a model that can answer questions from images/pdfs. To use it, simply upload your image or pdf and type a question and click 'submit', or click one of the examples to load them. Read more at the links below."
tochange_article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2212.02623' target='_blank'>Unifying Vision, Text, and Layout for Universal Document Processing</a> | <a href='https://github.com/microsoft/UDOP' target='_blank'>Github Repo</a></p>"

interface = gr.Interface(fn=answer_question, 
                         inputs=[image, question], 
                         outputs=answer, 
                         examples=examples, 
                         title=title,
                         description=description,
                         article=tochange_article)
interface.launch(debug=True)