import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import re

st.set_page_config(page_title="Customer Support", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Customer Support Chatbot")
st.caption("Powered by Groq & RAG")

# Replace "YOUR_API_KEY_HERE" with your actual Groq API key
GROQ_API_KEY = "YOUR_API_KEY_HERE"  
client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv.xls')
    df = df[['instruction', 'response']].dropna().drop_duplicates()
    return df

@st.cache_resource
def build_index(df, _encoder):
    embeddings = _encoder.encode(df['instruction'].tolist(), show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

encoder = load_encoder()
df = load_data()
index = build_index(df, encoder)

def retrieve(query, k=3):
    query_vec = encoder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return df.iloc[I[0]]

def generate_answer(query, context):
    prompt = f"""
    You are a STRICT, highly professional, and formal customer support agent. 
    Your ONLY job is to assist customers using the provided context.

    CRITICAL RULES - READ CAREFULLY:
    1. Answer the user's question using ONLY the provided context below.
    2. IF the user asks about topics OUTSIDE the context (e.g., writing code, general knowledge, math, cooking, geography, etc.), you MUST refuse to answer. Reply EXACTLY with: "I apologize, but I am a customer support assistant and can only assist you with inquiries related to our services and your orders."
    3. DO NOT invent, hallucinate, or provide outside information under ANY circumstances, even if you know the answer to the user's question.
    4. REWRITE the context in a highly professional, formal, and polite tone. REMOVE any slang or casual phrases.
    5. STRICTLY limit your answer to a maximum of 3 sentences. Do not use long lists.
    6. Do NOT include generic placeholders.

    Context:
    {context}

    User Question: {query}
    
    Response:
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-8b-instant", 
        temperature=0.0, 
        max_tokens=200,  
    )
    return chat_completion.choices[0].message.content.strip()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Customer Support Assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching records and thinking..."):
            try:
                retrieved_data = retrieve(prompt, k=3)
                context = ""
                for _, row in retrieved_data.iterrows():
                    context += f"- Customer asked: {row['instruction']}\n  Support replied: {row['response']}\n\n"
                
                answer = generate_answer(prompt, context)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")