BUCKET_NAME = "lloyds-genai24lon-2701-bucket"
PROJECT_ID = "lloyds-genai24lon-2701"
PREFIX = 'sample_50/'
MODEL_NAME = 'text-bison-32k'

from search import SemanticSearch 

def run_semantic_search(): 
    
    chain = SemanticSearch(project_id=PROJECT_ID, 
                           bucket_name = BUCKET_NAME,
                           prefix = PREFIX,
                           model_name = MODEL_NAME).get_semantic_chain()
    
    return chain 



if __name__=='main': 
    
    chain = run_semantic_chain()
    print(type(chain))