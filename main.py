import streamlit as st
import torch
import json
import data_processor

def predict(text):
    model_ei = torch.load("model/modelei.pth")
    model_ns = torch.load("model/modelns.pth")
    model_ft = torch.load("model/modelft.pth")
    model_jp = torch.load("model/modeljp.pth")
    with open('word_index.json') as json_file:
        vocab = json.load(json_file)

    extrovert, introvert = data_processor.predict(text, model_ei, 50, vocab)
    return extrovert, introvert

if __name__ == '__main__':
    # giving the webpage a title
    st.title("MBTI Type Prediction")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:blue;padding:13px">
    <h1 style ="color:black;text-align:center;">MBTI Type Classifier</h1>
    </div>
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    text = st.text_input("Insert Any Text", "")
    result = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    extr = 0
    intr = 0
    if st.button("Predict"):
        extr, intr = predict(text)
    st.success('The output is {}, {}'.format(extr, intr))
