import streamlit as st 
from pandasai.llm.openai import OpenAI #type: ignore
import pandas as pd
from pandasai import SmartDataframe #type: ignore
from pandasai import Agent 
from dotenv import load_dotenv
import os
from PIL import Image

st.title("Insight Analyzer")

openai_api_key= st.text_input("Enter OpenAI API key", type="password")
llm = OpenAI(api_token=openai_api_key)

uploader_file = st.file_uploader("Upload a CSV file", type= ["csv"])

def is_image(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write(data.head(3))

    df = SmartDataframe(data, config={"llm": llm})
    agent = Agent(data, config={"llm": llm, "open_charts":"False"})

    prompt = st.text_area("Enter your prompt:")
    system_prompt= "Decide whether this needs a graph. If it does determine what type of graph (i.e. line/scatter/bar/pie etc). then generate the graph according to the requirements. we may not always need you to generate a graph. if a graph is not required just generate an written output. If you are generating a graph also include a breif description"
    full_prompt= "Main query: " + prompt + "Instructions: " + system_prompt

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                graph = df.chat(full_prompt)
                if is_image(graph): 
                    st.image(graph)
                
                response= agent.chat(prompt + "Instructions: we don't generate graphs even if it is asked for")
                if not is_image(response):
                    st.write(response)
                
                with st.expander("See Explanation: "):
                    st.write(agent.explain())
                
        else:
            st.warning("Please enter a prompt!")