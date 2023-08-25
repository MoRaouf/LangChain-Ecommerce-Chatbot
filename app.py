import os
import pandas as pd

# Qdrant
import qdrant_client
from qdrant_client.http import models
from langchain.vectorstores import Qdrant


# LLM
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

# Chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# Streamlit
import streamlit as st
from streamlit_chat import message

# from getpass import getpass
# from uuid import uuid4

from dotenv import load_dotenv


#===================================================================================
#--------------------------------------- Keys --------------------------------------
#===================================================================================

# load necessary keys from `.env` file if you will NOT provide it through the app
from dotenv import load_dotenv
load_dotenv()  

os.environ['QDRANT_URL'] = 'https://77fd7f57-43ed-4636-ba19-c70f60736415.us-east-1-0.aws.cloud.qdrant.io:6333'

# unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"E-commerce Chatbot - Streamlit"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

#===================================================================================
#------------------------------------- Dataset -------------------------------------
#===================================================================================

# df = pd.read_csv("home_depot_data.csv", index_col=0)
# cols_to_keep = ['url', 'title', 'description', 'brand', 'price', 'currency', 'availability']
# df_2 = df[cols_to_keep]

# # Metadata
# metadata = df_2.to_dict(orient='index')
# # Text data that will be embedded and converted to vectors
# texts = [v['title'] for k, v in metadata.items()]
# # Product metadata that we'll store along our vectors
# payloads = list(metadata.values())

#===================================================================================
#-------------------------------------- Qdrant -------------------------------------
#===================================================================================

# Qdrant Client
client = qdrant_client.QdrantClient(
    url = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY")
    )

# Embeddings
embeddings = OpenAIEmbeddings()

collection_name = "ecommerce"

# Create Vector Store
vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )

#===================================================================================
#------------------------------------- Prompts -------------------------------------
#===================================================================================

question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_template)

# Chat LLM for question generation & Q&A
model_name = "gpt-3.5-turbo"
temperature = 0
llm_chat = ChatOpenAI(model=model_name, temperature=temperature)

# Chain for question generation
question_generator = LLMChain(llm=llm_chat, prompt=CONDENSE_QUESTION_PROMPT)

# Chat Prompt
system_template = """
You are a friendly, conversational retail shopping assistant. Use the following context including product names,
descriptions, and keywords to show the shopper whats available, help find what they want, and answer any questions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template= """Question: {question}"""

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Inject instructions into the prompt template.
# human_message_prompt = HumanMessagePromptTemplate(
#     prompt=PromptTemplate(
#         template=human_template,
#         input_variables=["question"],
#         partial_variables={"format_instructions": parser.get_format_instructions()}    
#         )
#     )

# Chat Prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Chain for Q&A
answer_chain = load_qa_chain(llm_chat,
                             chain_type="stuff",
                             prompt=chat_prompt)

#===================================================================================
#--------------------------- ConversationalRetrievalChain --------------------------
#===================================================================================

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Chain
qa = ConversationalRetrievalChain(
    retriever = vectorstore.as_retriever(),
    question_generator = question_generator,
    combine_docs_chain = answer_chain,
    memory=memory
)


# query = "I'm looking for a Hoodie"
# result = qa({"question": query})

# print(result['answer'])



#===================================================================================
#------------------------------------ Streamlit ------------------------------------
#===================================================================================

# Title
st.title("LangChain E-Commerce Chatbot")

# st.info('You can ask the Chatbot using audio or text.', icon="ℹ️")
st.markdown("***")

# ------------------------------------------------------------------------------------------

# Generate lists for human and ai.
if 'ai' not in st.session_state:
    st.session_state['ai'] = ["I'm your Assistant, How may I help you?"]

if 'human' not in st.session_state:
    st.session_state['human'] = ["Hi"]


# Layout of input & chat containers
chat_container = st.container()
st.markdown("***")
input_container = st.container()

# ------------------------------------------------------------------------------------------

# Submit button
submit_button = st.button('Submit')

# Response output
def generate_response(user_query):
    response = qa({"question": user_query})
    return response['answer']

with input_container:
    user_input = st.text_input("How can the assistant help you:")

# Conditional display of AI generated responses as a function of user provided prompts
with chat_container:
    if submit_button and user_input:
        # with st.spinner('Loading...'):
        response = generate_response(user_input)
        st.session_state.human.append(user_input)
        st.session_state.ai.append(response)
        
    if st.session_state['ai']:
        for i in range(len(st.session_state['ai'])):
            message(st.session_state['human'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["ai"][i], key=str(i))


# Example user input queries

# I want to renovate my living room. What material would I need?
# Which furniture you have in stock? I want something modern
# Give me more details about the coffee table