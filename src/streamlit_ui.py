import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from agent_build import create_agent


@st.cache_resource
def load_chain():
    return create_agent()

def generate_response(chain):
    return chain.invoke({"input": st.session_state.messages[-1], "chat_history": st.session_state.messages}, {"callbacks": [StreamlitCallbackHandler(st.container())]})

def main():
    chain = load_chain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
        with st.chat_message("assistant"):
            response = generate_response(chain)
            st.write(response["output"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["output"]
            }
        )

if __name__ == "__main__":
    main()