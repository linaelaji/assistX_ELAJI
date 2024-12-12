import pandas as pd
import numpy as np
import sqlite3
from sentence_transformers import util
import requests
from groq import Groq
import json
import logging
import re

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data():
    """Load and prepare data from the SQLite database."""
    conn = sqlite3.connect('data/assistance.sqlite3')
    df = pd.read_sql_query("SELECT * FROM assistance", conn)
    conn.close()
    df['full_content'] = df.apply(
        lambda row: f'Titre: "{row["title"]}". Contenu: "{row["content"]}". Article: "{row["url"]}".', axis=1)
    return df


def generate_embeddings(df):
    """Generate embeddings via ollama (running locally) for each document using an API and save them."""

    headers = {
        "Content-Type": "application/json",
    }
    embeddings = []
    for content in df['full_content'].tolist():

        data = {'model': 'nomic-embed-text', 'prompt': content}
        response = requests.post(
            'http://localhost:11434/api/embeddings', headers=headers, json=data)
        embedding = json.loads(response.text)

        if isinstance(embedding["embedding"], list) and len(embedding["embedding"]) == 768:
            embeddings.append(np.array(embedding["embedding"]))
            print("Embedding...")
        else:
            print(
                f"Received malformed embedding of length ")
    np.save('data/embeddings.npy', np.array(embeddings, dtype=object))


def load_embeddings():
    """Load embeddings from the saved file."""
    return np.load('data/embeddings.npy', allow_pickle=True)


def search_similar_documents(df, query, embeddings, top_k=5, threshold=0.42):
    """Search for the most similar documents in the dataframe using an API for embedding and hybrid scoring with a threshold."""
    headers = {
        "Content-Type": "application/json",
    }
    data = {'model': 'nomic-embed-text', 'prompt': query}
    response = requests.post(
        'http://localhost:11434/api/embeddings', headers=headers, json=data)
    embedding = json.loads(response.text)

    if isinstance(embedding["embedding"], list) and len(embedding["embedding"]) == 768:
        query_emb = np.array(embedding["embedding"], dtype=np.float32)
    else:
        print(f"Invalid or malformed query embedding received")
        return []

    if embeddings.dtype == object:
        embeddings = np.vstack(embeddings.tolist()).astype(np.float32)
    else:
        embeddings = embeddings.astype(np.float32)
    dense_scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    df['dense_score'] = dense_scores.numpy()

    filtered_documents = df[df['dense_score'] >= threshold]
    top_documents = filtered_documents.sort_values(
        by='dense_score', ascending=False).head(top_k)
    top_docs_data = [
        {
            "content": row['full_content'],
            "dense_score": row['dense_score'],
        } for index, row in top_documents.iterrows()
    ]
    return top_docs_data


def prepare_prompt(question, similar_docs, conversation_history):
    if similar_docs:
        docs_str = '\n'.join([doc['content'] for doc in similar_docs])
    else:
        docs_str = ""

    history_str = '\n'.join(
        [f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    if not conversation_history:
        prompt = f"""[INST]
        Tu es ProXigen, un assistant virtuel développé pour Free. 

        Question posée par le client : "{question}"

        Documents disponibles pour cette question : "{docs_str}"

        Réponse proposée :
        [/INST]
        """
    else:
        prompt = f"""[INST]
        Tu es toujours ProXigen, l'assistant virtuel de Free, 

        Question "{question}"

        Documents pour répondre question : "{docs_str}"

        Historique de la conversation : "{history_str}"

        Réponse proposée :
        [/INST]
        """
    return prompt.strip()


def query_with_ollama_api(question, conversation_history, similar_documents):
    """Query the model hosted on local ollama with robust streaming support, handling JSON fragments efficiently, with debug."""
    prompt = prepare_prompt(question, similar_documents, conversation_history)


def query_with_groq_api(question, api_key, conversation_history, similar_documents):
    """Query llama3 from groq"""

    prompt = prepare_prompt(question, similar_documents, conversation_history)

    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": f"{prompt}"
            }
        ],
        temperature=2,
        max_tokens=30,
        top_p=1,
        stream=False,
        stop=None,
    )
    print(type(chat_completion.choices[0].message.content))
    return chat_completion.choices[0].message.content
