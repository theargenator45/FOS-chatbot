BUCKET_NAME = "lloyds-genai24lon-2701-bucket"
PROJECT_ID = "lloyds-genai24lon-2701"
PREFIX = 'sample_50/'
MODEL_NAME = 'text-bison-32k'

from search import SemanticSearch 
from google.cloud import storage

  
def get_public_url(url): 
    
    components = url.split("/")
    bucket_name = components[2]
    file_name = components[3]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
        
    return blob.public_url

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


def display_results_source(result): 
    
    info = result['result'], 
    source_url = get_public_url(result['source_documents'][0].metadata['source'])
    file_url = result['source_documents'][1].metadata['source']
  
    
    return info, source_url, file_url



# if __name__=='__main__':
#     retriever = get_retriever()
    
#     chain = get_semantic_chain(retriever)
    
#     result = chain({"query": "what is the context of the first page"})
    
#     info, source_url, file = display_results_source(result)
    
#     print(info, source_url, file)

