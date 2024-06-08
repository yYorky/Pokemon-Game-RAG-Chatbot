import streamlit as st
import os

import pandas as pd
import chardet
import io
import re
from datasets import Dataset

import time
import pandas as pd
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.llms import Ollama
# from langchain_google_vertexai import ChatVertexAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from ragas.evaluation import evaluate
from ragas.metrics import (Faithfulness, 
                           AnswerRelevancy, 
                           ContextPrecision, 
                           ContextRelevancy,  
                           ContextRecall,
                           ContextEntityRecall, 
                           AnswerSimilarity, 
                           AnswerCorrectness)

# Import HTML templates for chat messages
from htmlTemplates import css, bot_template, user_template

# Load environment variables from .env file
load_dotenv()

# Load API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Load the evaluation dataset with encoding handling and debug info
def load_evaluation_data(uploaded_file):
    try:
        # Read the file content to detect encoding
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']
        
        st.sidebar.write(f"Detected file encoding: {detected_encoding}")
        
        # Convert the raw data to a string and read into a DataFrame
        content = raw_data.decode(detected_encoding)
        uploaded_file.seek(0)  # Reset file pointer to the beginning
        data = pd.read_csv(io.StringIO(content))
        
        st.sidebar.write("File content successfully read:")
        st.sidebar.write(data.head())  # Display the first few rows for debugging
        
        if data.empty:
            st.sidebar.error("The uploaded file is empty. Please upload a valid CSV file.")
            return None
        if not {'question', 'ground_truth'}.issubset(data.columns):
            st.sidebar.error("The uploaded CSV file must contain 'question' and 'ground_truth' columns.")
            return None
        dataset = Dataset.from_pandas(data)
        return dataset
    except Exception as e:
        st.sidebar.error(f"An error occurred while reading the file: {e}")
        return None

# Function for document embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings_initialized = False
        st.sidebar.write("Initializing embeddings...")

        if st.session_state.embedding_choice == 'OpenAI':
            st.session_state.embeddings = OpenAIEmbeddings()
        elif st.session_state.embedding_choice == 'Ollama':
            st.session_state.embeddings = OllamaEmbeddings(model="llama3")
        elif st.session_state.embedding_choice == 'GoogleGenerativeAI':
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        st.sidebar.write("Embeddings initialized.")

        st.sidebar.write("Loading documents from PDF directory...")
        st.session_state.loader = PyPDFDirectoryLoader("./groq/pokemon guide")  # Data Ingestion from PDF folder
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.sidebar.write(f"{len(st.session_state.docs)} documents loaded.")

        st.sidebar.write("Splitting documents into chunks...")
        chunk_size = st.session_state.chunk_size
        chunk_overlap = st.session_state.chunk_overlap
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.sidebar.write(f"{len(st.session_state.final_documents)} chunks created.")

        st.sidebar.write("Creating vector embeddings...")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.sidebar.write("Vector embeddings created.")
        st.session_state.embeddings_initialized = True

# Function to get the conversational retrieval chain
def get_conversation_chain(vectorstore, model_name):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, max_length=st.session_state.conversational_memory_length, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True  # Ensure source documents are returned
    )
    return conversation_chain

# Function to save evaluation results to CSV
def save_evaluation_results():
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        df = pd.DataFrame(st.session_state.evaluation_results)
        df.to_csv("evaluation_results.csv", index=False)
        st.sidebar.download_button(
            label="Download evaluation results",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='evaluation_results.csv',
            mime='text/csv'
        )
    else:
        st.sidebar.write("No evaluation results to save.")

# Initialize session state for evaluation results
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

# Add customization options to the sidebar
st.sidebar.title('Customization')
model = st.sidebar.selectbox(
    'Choose a model',
    ['llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
    key='model_choice',  # Unique key for the selectbox
)
st.session_state.embedding_choice = st.sidebar.selectbox(
    'Choose embedding type',
    ['GoogleGenerativeAI', 'OpenAI', 'Ollama'],
    key='embedding_choice_main'  # Unique key for the selectbox in the main sidebar
)
st.session_state.conversational_memory_length = st.sidebar.slider('Conversational memory length:', 0, 10, value=0)
st.session_state.chunk_size = st.sidebar.slider('Chunk size:', 1000, 8000, value=4000, step=500)
st.session_state.chunk_overlap = st.sidebar.slider('Chunk overlap:', 0, 1000, value=500, step=100)



# Add a text area for the prompt
prompt = st.sidebar.text_area("Enter a prompt for the LLM:", key="prompt")

if st.sidebar.button("Documents Embedding"):
    vector_embedding()
    st.sidebar.write("Vector Store DB Is Ready")
    st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectors, model)  # Initialize the conversation chain

# File uploader for evaluation dataset
uploaded_file = st.sidebar.file_uploader("Upload evaluation dataset (CSV)", type="csv")
if uploaded_file is not None:
    st.session_state.evaluation_data = load_evaluation_data(uploaded_file)

# Button to save evaluation results
if st.sidebar.button("Save Evaluation Results"):
    save_evaluation_results()

# Displaying a GIF
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/EXemST2.gif" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Professor Chatgroq for Pokemon Scarlet & Violet")

# Session state variable
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def clean_text(text):
    # Remove excessive newlines and unwanted characters
    return re.sub(r'\s+', ' ', text).strip()

def handle_userinput():
    user_question = st.session_state.user_question
    if not st.session_state.get("embeddings_initialized", False):
        st.write("Please initialize the embeddings first.")
        return
    
    if "conversation_chain" not in st.session_state:
        st.write("Please initialize the conversation chain first.")
        return

    start = time.process_time()
    
    # Invoke the retrieval chain with the prompt included
    response = st.session_state.conversation_chain({'question': f"{prompt} {user_question}", 'chat_history': st.session_state.chat_history})
    ai_response = response['answer']
    source_documents = response.get('source_documents', [])

    # Save the user input and AI response to the chat history
    st.session_state.chat_history.append({'human': user_question, 'AI': ai_response})

    st.session_state.user_question = ""

    display_chat_history()

    # Store response for document similarity search
    st.session_state.response = response

    # Prepare context in the correct format
    context_texts = [[clean_text(doc.page_content) for doc in source_documents]]

    # RAGAS Evaluation
    if uploaded_file is not None and st.session_state.evaluation_data is not None:
        evaluation_data = st.session_state.evaluation_data
        filtered_data = evaluation_data.filter(lambda x: x['question'] == user_question)
        
        if len(filtered_data) > 0:
            ground_truth = filtered_data['ground_truth'][0]
            metrics = [
                Faithfulness(), 
                AnswerRelevancy(), 
                ContextPrecision(), 
                ContextRelevancy(),  
                ContextRecall(),
                ContextEntityRecall(), 
                AnswerSimilarity(), 
                AnswerCorrectness()
            ]
            evaluation_data = Dataset.from_dict({
                'question': [user_question],
                'ground_truth': [ground_truth],
                'answer': [ai_response],
                'contexts': context_texts,
            })

            # langchain_llm = ChatVertexAI(model="gemini-pro")
            # langchain_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            
            evaluation_result = evaluate(evaluation_data, 
                                         metrics=metrics,
                                        #  llm=langchain_llm, 
                                        #  embeddings=langchain_embeddings
                                         )
            
            # Store evaluation result
            result_entry = {
                'question': user_question,
                'ground_truth': ground_truth,
                'answer': ai_response,
                'contexts': context_texts,
                'faithfulness': evaluation_result['faithfulness'],
                'answer_relevancy': evaluation_result['answer_relevancy'],
                'context_precision': evaluation_result['context_precision'],
                'context_relevancy': evaluation_result['context_relevancy'],
                'context_recall': evaluation_result['context_recall'],
                'context_entity_recall': evaluation_result['context_entity_recall'],
                'answer_similarity': evaluation_result['answer_similarity'],
                'answer_correctness': evaluation_result['answer_correctness'],
            }
            st.session_state.evaluation_results.append(result_entry)
            
            st.write("RAGAS Evaluation Results:")
            st.write(evaluation_result)
        else:
            st.write("Question not found in the evaluation dataset. RAGAS evaluation skipped.")
    
    else:
        st.write("Please upload an evaluation dataset.")


def display_chat_history():
    st.write(css, unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", message['human']), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message['AI']), unsafe_allow_html=True)

def main():
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.text_input("Ask Professor Chatgroq any question about Pokemon Scarlet & Violet:", key="user_question", on_change=handle_userinput)

    # With a streamlit expander
    if 'response' in st.session_state:
        with st.expander("Document Similarity Search"):
            if "source_documents" in st.session_state.response:
                for i, doc in enumerate(st.session_state.response["source_documents"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No source documents found in the response.")

if __name__ == '__main__':
    main()