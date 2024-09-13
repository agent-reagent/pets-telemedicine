import streamlit as st
import os
from groq import Groq
from key import API_KEY


client = Groq(
    api_key= API_KEY ,
)
st.title("My First Streamlit App")


st.header("Welcome to My Streamlit App!")

u_input = st.text_input("Enter input:")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": u_input,
        }
    ],
    model="llama3-8b-8192",
)




if st.button("Submit"):
    st.write(f"LLM Response")
    st.write(chat_completion.choices[0].message.content)


# Show a footer message
st.write("Thank you for using my app!")

