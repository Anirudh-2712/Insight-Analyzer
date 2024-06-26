import streamlit as st 
from pandasai.llm.openai import OpenAI
import pandas as pd
from pandasai import SmartDataframe, Agent
import io

st.title("Insight Analyzer")

openai_api_key = st.text_input("Enter OpenAI API key", type="password")

if openai_api_key:
    llm = OpenAI(api_token=openai_api_key)

    uploader_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploader_file is not None:
        data = pd.read_csv(uploader_file, dayfirst=True)
        st.write(data.head(3))

        df = SmartDataframe(data, config={"llm": llm})
        agent = Agent(data, config={"llm": llm, "open_charts": "False"})

        prompt = st.text_area("Enter your prompt:")
        system_prompt = "Decide whether this needs a graph. If it does determine what type of graph (i.e. line/scatter/bar/pie etc). then generate the graph according to the requirements. we may not always need you to generate a graph. if a graph is not required just generate a written output. If you are generating a graph also include a brief description."
        full_prompt = "Main query: " + prompt + " Instructions: " + system_prompt

        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    try:
                        graph = df.chat(full_prompt)
                        if "temp_chart" in str(graph):
                            st.image(graph)

                        response = agent.chat(prompt + " Instructions: we don't generate graphs even if it is asked for")
                        if "temp_chart" not in str(response):
                            st.write(response)

                        with st.expander("See Explanation:"):
                            st.write(agent.explain())
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a prompt!")
else:
    st.warning("Please enter your OpenAI API key!")
