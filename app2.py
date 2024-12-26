import streamlit as st
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import httpx, base64
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

#%%writefile app.py
import locale, os
import streamlit as st
locale.getpreferredencoding = lambda: "UTF-8"
import tempfile

import yaml

# MULTIMODAL INPUTS OPEN AI

# https://python.langchain.com/docs/integrations/chat/google_generative_ai/-GEMINI-WITH-LANGCHAIN

with open('gemini_key.yml') as f:
    api_creds = yaml.safe_load(f)
    
os.environ["GOOGLE_API_KEY"] = api_creds['gemini_key']

with open('chatgpt_api_credentials.yml', 'r') as file:
    api_creds = yaml.safe_load(file)

os.environ['OPENAI_API_KEY'] = api_creds['openai_key']

def get_content(image_path):
        """
        Fetches the image from the given path, generates a caption and summary using a language model, 
        and returns the generated text.
        """
        #return "image is about studying for exams"
        #st.text(os.path.isfile(image_path))
        
        if os.path.isfile(image_path):
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
                return image_data
        else:
            try:
                image_data = base64.b64encode(httpx.get(image_path).content).decode("utf-8")
                image_data.raise_for_status() 
                return image_data

            except httpx.RequestError as e:
                print(f"Error fetching image: {e}")
                return "Failed to fetch image."
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return "An error occurred while processing the image."

openai_model = ChatOpenAI(model="gpt-4o",temperature=0, streaming = True)

SYS_PROMPT = """
You are a helpful AI Assistant. The user will upload an image.
Your job is to use information about the image to answer the user's questions
about the image. Any other questions should not be entertained and you should
reply politely saying you only answer questions about the uploaded image.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

streamlit_msg_history = StreamlitChatMessageHistory()

def main():
    """
    The main function that handles image upload, user interaction,
    and LLM-based question answering.
    """
    
    st.set_page_config(page_title = 'AI Assistant')
    st.title('Image Q&A Chatbot')
    st.write("This assistant specializes in answering questions about an uploaded image. It uses information from the image to provide accurate and informative responses.")

    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        image_data = get_content(temp_filepath)
        st.write("Image uploaded!")
        st.image(temp_filepath, caption="Uploaded Image")

        conversation_history = []  # Track conversation history

        # Initialize session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from session state
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        #overall_history = ""
        if prompt := st.chat_input("What is your question?"):
            # Add user message to session state
            #overall_history = overall_history + f"User: {prompt}\n" 
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},},
                        ]
                )
                    
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                response = openai_model.invoke([message], stream = True)
                
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                #overall_history = overall_history + f"LLM Assistant: {response.content}\n"
                        
                with st.chat_message("assistant"):
                    st.markdown(response.content)
                        #st.write(response.content)
                #conversation_history.append(prompt)
                #conversation_history.append(response.content)
                #streamlit_msg_history.update_history(conversation_history)  # Update history (optional)
            except Exception as e:
                print(f"Error: {e}")
                st.write(e)
                st.write("An error occurred while processing your request.")

            # Display the updated conversation history after each user input
            for msg in streamlit_msg_history.messages:
                st.chat_message(msg.type).write(msg.content)

            # Update counter for unique keys
            if 'counter' not in st.session_state:
                st.session_state['counter'] = 0
            st.session_state['counter'] += 1 

if __name__ == "__main__":
    main()
