import pandas as pd
import numpy as np
import faiss
import sqlite3
from sentence_transformers import util
import requests
from groq import Groq
import json
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data():
    """Load and prepare data from the SQLite database."""
    conn = sqlite3.connect('data/assistance.sqlite')
    df = pd.read_sql_query("SELECT * FROM assistance", conn)
    conn.close()
    df['full_content'] = df.apply(
        lambda row: f'Titre: "{row["title"]}". Contenu: "{row["content"]}". Article: "{row["url"]}".', axis=1)
    return df


def generate_embeddings(df):
    """Génère et sauvegarde les embeddings avec FAISS"""
    headers = {"Content-Type": "application/json"}
    embeddings = []
    
    for content in df['full_content'].tolist():
        data = {'model': 'nomic-embed-text', 'prompt': content}
        response = requests.post(
            'http://localhost:11434/api/embeddings', 
            headers=headers, 
            json=data
        )
        embedding = json.loads(response.text)
        
        if isinstance(embedding["embedding"], list):
            embeddings.append(np.array(embedding["embedding"]))
            print("Embedding...")
    
    # Convertir en array numpy
    embeddings_array = np.array(embeddings).astype('float32')
    
   

   
    return embeddings_array

def load_embeddings():
    """Load embeddings from the saved file."""
    return np.load('data/embeddings.npy', allow_pickle=True)


def search_similar_documents(df, query, index, top_k=5):
    """Recherche les documents similaires avec FAISS"""
    try:
        headers = {"Content-Type": "application/json"}
        data = {'model': 'nomic-embed-text', 'prompt': query}
        response = requests.post(
            'http://localhost:11434/api/embeddings', 
            headers=headers, 
            json=data
        )
        query_embedding = json.loads(response.text)
        # print("Taille de l'embedding du query :", len(query_embedding["embedding"]))
        
        if isinstance(query_embedding["embedding"], list):
            query_vector = np.array([query_embedding["embedding"]]).astype('float32')
            
            # Recherche avec FAISS
            distances, indices = index.search(query_vector, top_k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(df):  # vérification de sécurité
                    score = 1 / (1 + distance)  # convertir distance en score de similarité
                    results.append({
                        "content": df.iloc[idx]['full_content'],
                        "dense_score": float(score)
                    })
            
            return results
    except Exception as e:
        print(f"Erreur lors de la recherche FAISS: {e}")
        return []


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
    try:
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
            temperature=0.3,
            max_tokens=500,
            top_p=1,
            stream=False,
            stop=None,
        )
        print(type(chat_completion.choices[0].message.content))
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Erreur lors de la requête Groq: {e}")
        return "Je rencontre actuellement un problème, veuillez réessayer plus tard."