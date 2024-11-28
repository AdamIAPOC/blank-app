import streamlit as st
from dotenv import load_dotenv
import os
import time
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Configuration de la page avec style personnalisé
st.set_page_config(page_title="CSRD/ESRS Assistant Adamantia", layout="wide", page_icon="📄")

# Style CSS personnalisé
st.markdown("""
<style>
.stTextInput > div > div > input {
    border: 2px solid #3498db;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
}
.response-box {
    background-color: #f0f4f8;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
}
.question-section {
    background-color: #e6f2ff;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

def calculate_cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def check_response_relevance(question, response, retrieved_docs, threshold=0.5):
    """
    Vérifie la pertinence de la réponse par rapport aux documents et à la question.
    
    Args:
        question (str): La question posée
        response (str): La réponse générée
        retrieved_docs (list): Documents récupérés
        threshold (float): Seuil de similarité
    
    Returns:
        bool: Indique si la réponse est pertinente
    """
    try:
        embeddings = OpenAIEmbeddings()
        
        # Embedding de la question
        question_embedding = embeddings.embed_query(question)
        
        # Embedding de la réponse
        response_embedding = embeddings.embed_query(response)
        
        # Embedding des documents
        doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in retrieved_docs]
        
        # Calcul des similarités
        response_doc_similarities = [calculate_cosine_similarity(response_embedding, doc_emb) 
                                     for doc_emb in doc_embeddings]
        
        # Moyenne des similarités
        avg_similarity = np.mean(response_doc_similarities)
        
        return avg_similarity >= threshold
    
    except Exception as e:
        st.error(f"Erreur lors de la vérification de pertinence : {e}")
        return False

# Reste de votre code d'initialisation identique
def initialize_application():
    load_dotenv()
    API_KEY = os.getenv('OPENAI_API_KEY')

    # Initialisation du modèle
    model = ChatOpenAI(api_key=API_KEY, model='gpt-3.5-turbo', temperature=0.2)

    # Chargement des documents
    documents = []
    pdf_folder = "Documentation_CSRD"
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    # Préparation des documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pages = splitter.split_documents(documents)

    # Base vectorielle
    try:
        vector_storage = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    except:
        vector_storage = FAISS.from_documents(pages, OpenAIEmbeddings())
        vector_storage.save_local("faiss_index")

    # Configuration du retriever
    retriever = vector_storage.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Template de prompt
    question_template = """
    Tu es un expert en réglementation CSRD/ESRS. 
    Contexte: {context}
    Question: {question}
    
    Réponds de manière professionnelle et précise. 
    Si tu ne peux pas répondre avec certitude à partir du contexte, dis-le clairement.
    """

    prompt = PromptTemplate.from_template(template=question_template)
    
    # Chaîne de traitement
    def custom_invoke(input_question):
        # Récupération des documents
        retrieved_docs = retriever.get_relevant_documents(input_question)
        
        # Préparation du contexte
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Génération de la réponse
        response = model.invoke(
            prompt.format(
                context=context, 
                question=input_question
            )
        ).content
        
        # Vérification de pertinence
        if not check_response_relevance(input_question, response, retrieved_docs):
            return "Je suis désolé, mais je n'ai pas trouvé d'informations suffisamment précises dans mes documents pour répondre à cette question avec certitude."
        
        return response

    return custom_invoke

# Configuration principale
def main():
    st.title("🔍 Assistant CSRD/ESRS")
    
    # Initialisation de la chaîne de traitement
    if 'chain' not in st.session_state:
        st.session_state.chain = initialize_application()
    
    # Section de question stylisée
    st.markdown("<div class='question-section'>", unsafe_allow_html=True)
    question = st.text_input(
        "Posez votre question", 
        placeholder="Par exemple : Quels sont les principaux changements introduits par la CSRD ?",
        help="Formulez votre question sur la réglementation CSRD/ESRS"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if question:
        with st.spinner("Analyse en cours..."):
            # Génération de la réponse
            response = st.session_state.chain(question)
            
            # Affichage dans un cadre stylisé
            st.markdown("<div class='response-box'>", unsafe_allow_html=True)
            st.markdown("### 📝 Votre Question")
            st.write(question)
            st.markdown("### 💡 Réponse")
            st.write(response)
            st.markdown("</div>", unsafe_allow_html=True)

# Exécution principale
if __name__ == "__main__":
    main()