BUCKET_NAME = "lloyds-genai24lon-2701-bucket"
PROJECT_ID = "lloyds-genai24lon-2701"
PREFIX = 'sample_50/'
MODEL_NAME = 'text-bison-32k'

from search import SemanticSearch 

def get_retriever(): 
    
    retriever = SemanticSearch(project_id=PROJECT_ID, 
                           bucket_name = BUCKET_NAME,
                           prefix = PREFIX,
                           model_name = MODEL_NAME).embeddings_to_vector_db()
    
    return retriever

def get_semantic_chain(retriever): 
    
    chain = SemanticSearch(project_id=PROJECT_ID, 
                           bucket_name = BUCKET_NAME,
                           prefix = PREFIX,
                           model_name = MODEL_NAME).retrieve_semantic_chain(retriever)
    
    return chain



