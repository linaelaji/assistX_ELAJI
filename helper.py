import pandas as pd
import numpy as np
import faiss
import sqlite3
from sentence_transformers import util
import requests
from groq import Groq
import json
import logging
from typing import List, Dict
import torch
import requests
import json
from groq import Groq
import os


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



#HERE 1
def search_similar_documents(df, query, index, top_k=3, rerank_top_k=5):
    """
    Recherche les documents similaires avec FAISS et applique un reranking
    pour améliorer la pertinence des résultats
    """
    try:
        # Première phase: Recherche dense avec FAISS
        headers = {"Content-Type": "application/json"}
        data = {'model': 'nomic-embed-text:latest', 'prompt': query}
        response = requests.post(
            'http://localhost:11434/api/embeddings', 
            headers=headers, 
            json=data
        )
        query_embedding = json.loads(response.text)
    
        if isinstance(query_embedding["embedding"], list):
            query_vector = np.array([query_embedding["embedding"]]).astype('float32')
           
            # Recherche initiale avec FAISS - on récupère plus de résultats pour le rerankin
           
            distances, indices = index.search(query_vector, top_k)
            
            # Préparation des paires pour le reranking
            candidate_pairs = []
            candidates = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(df):
                    doc_content = df.iloc[idx]['full_content']
                    candidate_pairs.append([query, doc_content])
                    candidates.append({
                        "content": doc_content,
                        "dense_score": float(1 / (1 + distance)),
                    })
            # Deuxième phase: Reranking avec un modèle cross-encoder
            rerank_scores = rerank_candidates(candidate_pairs)
            print("rerank scores:", rerank_scores)
            # Combiner les scores et trier les résultats
            for candidate, rerank_score in zip(candidates, rerank_scores):
                # Combiner les scores (vous pouvez ajuster les poids)
                candidate["final_score"] = (
                    0.3 * candidate["dense_score"] + 
                    0.7 * rerank_score/10
                )
            
            # Trier par score final et prendre les top_k meilleurs résultats
            results = sorted(
                candidates, 
                key=lambda x: x["final_score"], 
                reverse=True
            )[:rerank_top_k]
            
            return results
            
    except Exception as e:
        print(f"Erreur lors de la recherche et du reranking: {e}")
        return []

def rerank_candidates(candidate_pairs: List[List[str]]) -> List[float]:
    """
    Réordonne les candidats en utilisant un modèle cross-encoder local
    """
    print("reranking candidate")
    try:
        print("Réordonne les candidats en utilisant un modèle cross-encoder local")
        # Utiliser l'API Ollama pour le reranking
        scores = []
        
        for query, doc in candidate_pairs:
            prompt = f"""[INST]
            Sur une échelle de 0 à 10, évalue la pertinence du document suivant par rapport à la question.
            
            Question: {query}
            Document: {doc}
            
            Réponds avec un json qui contient le resultat: 
            
            Exemple:
            {{
                'result':5
            }}
            [/INST]"""
            
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
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
            response_format={"type": "json_object"}
        )   
            result = chat_completion.choices[0].message.content
            print("result",result)
            result = json.loads(result)["result"]
            try:
                score = float(result)
                # Vérifier que le score est bien entre 0 et 1
                score = max(0, min(10, score))
            except ValueError:
                # En cas d'erreur de conversion, utiliser un score par défaut
                print("erreur dans rerank")
                score = 5
                
            scores.append(score)
            
        return scores
        
    except Exception as e:
        print(f"Erreur lors du reranking: {e}")
        # En cas d'erreur, retourner des scores neutres
        return [0.5] * len(candidate_pairs)
    #here fin
    

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
        Si tu utilise un document, donne le lien de l'article.

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



