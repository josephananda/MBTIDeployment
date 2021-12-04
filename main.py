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
    intuition, sensing = data_processor.predict(text, model_ns, 50, vocab)
    feeling, thinking = data_processor.predict(text, model_ft, 50, vocab)
    judging, perceiving = data_processor.predict(text, model_jp, 50, vocab)
    return extrovert, introvert, intuition, sensing, feeling, thinking, judging, perceiving

def get_explanation(types):
    if types == "INFP":
        return "INFPs are imaginative idealists, guided by their own core values and beliefs. To a Healer, possibilities are paramount; the realism of the moment is only of passing concern. They see potential for a better future, and pursue truth and meaning with their own individual flair."
    elif types == "INFJ":
        return "INFJs are creative nurturers with a strong sense of personal integrity and a drive to help others realize their potential. Creative and dedicated, they have a talent for helping others with original solutions to their personal challenges."
    elif types == "ENFJ":
        return "ENFJs are idealist organizers, driven to implement their vision of what is best for humanity. They often act as catalysts for human growth because of their ability to see potential in other people and their charisma in persuading others to their ideas. They are focused on values and vision, and are passionate about the possibilities for people."
    elif types == "ENFP":
        return "ENFPs are people-centered creators with a focus on possibilities and a contagious enthusiasm for new ideas, people and activities. Energetic, warm, and passionate, ENFPs love to help other people explore their creative potential."
    elif types == "INTJ":
        return "INTJs are analytical problem-solvers, eager to improve systems and processes with their innovative ideas. They have a talent for seeing possibilities for improvement, whether at work, at home, or in themselves."
    elif types == "ENTJ":
        return "ENTJs are strategic leaders, motivated to organize change. They are quick to see inefficiency and conceptualize new solutions, and enjoy developing long-range plans to accomplish their vision. They excel at logical reasoning and are usually articulate and quick-witted."
    elif types == "ENTP":
        return "ENTPs are inspired innovators, motivated to find new solutions to intellectually challenging problems. They are curious and clever, and seek to comprehend the people, systems, and principles that surround them. Open-minded and unconventional, Visionaries want to analyze, understand, and influence other people."
    elif types == "INTP":
        return "INTPs are philosophical innovators, fascinated by logical analysis, systems, and design. They are preoccupied with theory, and search for the universal law behind everything they see. They want to understand the unifying themes of life, in all their complexity."
    elif types == "ESFJ":
        return "ESFJs are conscientious helpers, sensitive to the needs of others and energetically dedicated to their responsibilities. They are highly attuned to their emotional environment and attentive to both the feelings of others and the perception others have of them. ESFJs like a sense of harmony and cooperation around them, and are eager to please and provide."
    elif types == "ESFP":
        return "ESFPs are vivacious entertainers who charm and engage those around them. They are spontaneous, energetic, and fun-loving, and take pleasure in the things around them: food, clothes, nature, animals, and especially people."
    elif types == "ISFJ":
        return "ISFJs are industrious caretakers, loyal to traditions and organizations. They are practical, compassionate, and caring, and are motivated to provide for others and protect them from the perils of life."
    elif types == "ISFP":
        return "ISFPs are gentle caretakers who live in the present moment and enjoy their surroundings with cheerful, low-key enthusiasm. They are flexible and spontaneous, and like to go with the flow to enjoy what life has to offer. ISFPs are quiet and unassuming, and may be hard to get to know. However, to those who know them well, the ISFP is warm and friendly, eager to share in life's many experiences."
    elif types == "ESTJ":
        return "ESTJs are hardworking traditionalists, eager to take charge in organizing projects and people. Orderly, rule-abiding, and conscientious, ESTJs like to get things done, and tend to go about projects in a systematic, methodical way."
    elif types == "ESTP":
        return "ESTPs are energetic thrillseekers who are at their best when putting out fires, whether literal or metaphorical. They bring a sense of dynamic energy to their interactions with others and the world around them. They assess situations quickly and move adeptly to respond to immediate problems with practical solutions."
    elif types == "ISTJ":
        return "ISTJs are responsible organizers, driven to create and enforce order within systems and institutions. They are neat and orderly, inside and out, and tend to have a procedure for everything they do. Reliable and dutiful, ISTJs want to uphold tradition and follow regulations."
    elif types == "ISTP":
        return "ISTPs are observant artisans with an understanding of mechanics and an interest in troubleshooting. They approach their environments with a flexible logic, looking for practical solutions to the problems at hand. They are independent and adaptable, and typically interact with the world around them in a self-directed, spontaneous manner."


if __name__ == '__main__':
    # giving the webpage a title
    #st.sidebar.button("Home")
    st.title("MBTI Type Predictor")
    select_pages = st.sidebar.selectbox(
        'Pages',
        ('Home', 'MBTI Predictor')
    )

    if select_pages == "Home":
        st.subheader("About This Project")
        st.markdown('<p style="color:Black;">The Myers-Briggs Type Indicator (MBTI) is a personality model developed by Katharine Cooks Briggs and Isabel Briggs Myers in 1940. The MBTI personality displays a combination of preferences from four different domains. As many as 80 percent of the top 500 companies and 89 percent of the top 100 companies in the United States use the MBTI personality test in the recruitment process. Generally, test takers need to answer about 50 to 70 questions and it is relatively expensive to know MBTI personality. To solve this problem, the researcher developed a personality classification system using the Convolutional Neural Network (CNN) methods and GloVe (Global Vectors for Word Representation) word embedding.</p>', unsafe_allow_html = True)
        st.text("")
        st.markdown('<p style="color:Black;">This project is intended for undergraduate thesis with the title of "Myers-Briggs Type Indicator (MBTI) Personality Model Classification in English Text using Convolutional Neural Network (CNN) Method".</p>', unsafe_allow_html = True)
        st.text("")
        st.markdown('<p style="color:Black;">Student Name: Joseph Ananda Sugihdharma</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:Black;">Supervisor Name: Dr.Eng. Fitra Abdurrachman Bachtiar, S.T., M.Eng.</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:Black;"><b>Last Updated: 04-Des-2021</b></p>', unsafe_allow_html=True)

    if select_pages == "MBTI Predictor":
        # the following lines create text boxes in which the user can enter
        # the data required to make the prediction
        st.subheader("Questions")
        st.markdown('<p style="color:Black;"><b>What did you usually do after the lecture ends?</b></p>',
                    unsafe_allow_html=True)
        text = st.text_area("Write your answer below", "")
        extrovert = 0
        introvert = 0
        intuition = 0
        sensing = 0
        feeling = 0
        thinking = 0
        judging = 0
        perceiving = 0
        if st.button("Predict"):
            extrovert, introvert, intuition, sensing, feeling, thinking, judging, perceiving = predict(text)
        # st.success(f"Extrovert: {extrovert:.2f}%, Introvert: {introvert:.2f}%")
        # st.success(f"Intuition: {intuition:.2f}%, Sensing: {sensing:.2f}%")
        # st.success(f"Feeling: {feeling:.2f}%, Thinking: {thinking:.2f}%")
        # st.success(f"Judging: {judging:.2f}%, Perceiving: {perceiving:.2f}%")
        if text != "":
            if extrovert > introvert:
                st.markdown(
                    f'<p style="color:Black;"><b>Extrovert: {extrovert:.2f}%</b>, Introvert: {introvert:.2f}%</p>',
                    unsafe_allow_html=True)
                a = "E"
            else:
                st.markdown(
                    f'<p style="color:Black;">Extrovert: {extrovert:.2f}%, <b>Introvert: {introvert:.2f}%</b></p>',
                    unsafe_allow_html=True)
                a = "I"

            if intuition > sensing:
                st.markdown(f'<p style="color:Black;"><b>Intuition: {intuition:.2f}%</b>, Sensing: {sensing:.2f}%</p>',
                            unsafe_allow_html=True)
                b = "N"
            else:
                st.markdown(f'<p style="color:Black;">Intuition: {intuition:.2f}%, <b>Sensing: {sensing:.2f}%</b></p>',
                            unsafe_allow_html=True)
                b = "S"

            if feeling > thinking:
                st.markdown(f'<p style="color:Black;"><b>Feeling: {feeling:.2f}%</b>, Thinking: {thinking:.2f}%</p>',
                            unsafe_allow_html=True)
                c = "F"
            else:
                st.markdown(f'<p style="color:Black;">Feeling: {feeling:.2f}%, <b>Thinking: {thinking:.2f}%</b></p>',
                            unsafe_allow_html=True)
                c = "T"

            if intuition > sensing:
                st.markdown(
                    f'<p style="color:Black;"><b>Judging: {judging:.2f}%</b>, Perceiving: {perceiving:.2f}%</p>',
                    unsafe_allow_html=True)
                d = "J"
            else:
                st.markdown(
                    f'<p style="color:Black;">Judging: {judging:.2f}%, <b>Perceiving: {perceiving:.2f}%</b></p>',
                    unsafe_allow_html=True)
                d = "P"

            types = a + b + c + d
            st.text("")
            st.markdown(f'<p style="color:Black; font-size:20px">Your final label is: <b>{types}</b></p>',
                        unsafe_allow_html=True)
            explanation = get_explanation(types)
            st.text("")
            st.markdown(f'<p style="color:Black;"><b>Explanation:</b></p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:Black;">{explanation}</p>', unsafe_allow_html=True)

