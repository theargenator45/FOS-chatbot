import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from test_build import create_agent
import re

logo = "lloyds-logo.webp"
st.set_page_config(page_icon=logo, page_title="FOS Chatbot")
st.title("FOS Complaints Chatbot")

@st.cache_resource
def load_chain():
    return create_agent()

def extract_url_and_embed(llm_response):
    doc_lookup_step = [x for x in llm_response["intermediate_steps"] if x[0].tool == "document_lookup"]
    if doc_lookup_step:
        url = re.search("https://(.*?).pdf",doc_lookup_step[-1][1])
        if url:
            url = url.group(0)
            html = f"The source document can be viewed [here]({url})"
            return html


def generate_response(chain):
    response = chain.invoke({"input": st.session_state.messages[-1], "chat_history": st.session_state.messages}, {"callbacks": [StreamlitCallbackHandler(st.container())]})
    #print(response["intermediate_steps"])
    url_html = extract_url_and_embed(response)
    if url_html:
        return response["output"] + "\n\n" + url_html

    return response["output"]

def main():
    chain = load_chain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=logo if message["role"] == "assistant" else None):
            st.write(message["content"])

    if user_message := st.chat_input():
        st.chat_message("user").write(user_message)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        ) 


    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar=logo):
            response = generate_response(chain)
            st.write(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

if __name__ == "__main__":
    main()