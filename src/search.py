from langchain_google_vertexai import VertexAI
from langchain import LLMChain
from langchain_community.document_loaders import GCSDirectoryLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter



CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
REQUESTS_PER_MINUTE = 50

class SemanticSearch(): 
    
    def __init__(self, project_id, bucket_name, prefix, model_name): 
        
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.prefix = prefix 
        self.model_name = model_name
        
    def initialize_llm(self): 
        return VertexAI(model_name = self.model_name, 
                        max_output_tokens = 256, 
                        temperature = 0.1, top_p = 0.8, 
                           top_k = 40, verbose = True)

        
    
    def load_documents_from_storage(self): 
        loader = GCSDirectoryLoader(project_name= self.project_id, 
                                  bucket = self.bucket_name, 
                                  prefix = self.prefix)
        return loader.load()
    
    def split_and_chunk(self, chunk_size, chunk_overlap, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                       chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(documents)
        return docs
    
    def get_embeddings(self,requests_per_minute):
        return VertexAIEmbeddings(requests_per_minute= REQUESTS_PER_MINUTE)
    
    def get_retriever(self,docs, embeddings):
        documents_vector_db = Chroma.from_documents(docs, embeddings)
        return documents_vector_db.as_retriever(search_type = 'mmr', search_kwargs = {"k":3})
    
    def embeddings_to_vector_db(self):
        
        documents = self.load_documents_from_storage()
        
        # Split and chunk documents
        docs = self.split_and_chunk(CHUNK_SIZE, CHUNK_OVERLAP, documents)
        
        #get embeddings 
        embeddings = self.get_embeddings(REQUESTS_PER_MINUTE)
        
        # Vector search retrieval
        
        retriever = self.get_retriever(docs, embeddings)
        
        return retriever
  

    def return_chain(self,llm, retriever): 
        return RetrievalQA.from_chain_type(llm = llm, 
                                           chain_type = "stuff", 
                                           retriever=retriever,
                                           return_source_documents=True)
    
    def retrieve_semantic_chain(self, retriever):
        
        
        # return _chain 
        
        llm = self.initialize_llm()
        
        semantic_chain = self.return_chain(llm, retriever)
        
        return semantic_chain
        
        
        
   

        
        
        
        
    