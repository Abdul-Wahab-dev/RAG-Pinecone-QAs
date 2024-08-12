from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import VectorDBQA
from lib.pinecone import PineconeQueries
from config.pinecone import PineconeClient

import os
class LangchainClient:
    pineconeClient = None
    def __init__(self):
        print(os.environ.get('OPENAI_KEY'),'os.environ.get()')
        self.embeding = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_KEY'), model='text-embedding-ada-002')
        self.pineconeClient = PineconeClient()
        self.openaiClient = ChatOpenAI(api_key=os.environ.get('OPENAI_KEY'), model='gpt-3.5-turbo-16k')
        self.pineconeQueries = PineconeQueries()
        
        
    def qaChain(self , *args , **kwargs):
        query = kwargs['query']
        vector_store = self.getVectorStore()
        chain = VectorDBQA.from_chain_type(llm=self.openaiClient, chain_type='stuff', vectorstore=vector_store)
        result = chain.run(query)
        return result
        
        
        
    def embedData(self,*args , **kwargs):
        data = [docs for docs in kwargs['docs']]
        getPineconeIndex = self.pineconeQueries.getIndex('temp')
        if len(data) > 0:
            vector_store = PineconeVectorStore(index_name='temp',embedding=self.embeding)
            vector_store.from_documents(documents=data,embedding=self.embeding, index_name="temp")
            return True
        return False
        
    
    def getVectorStore(self):
        getPineconeIndex = self.pineconeQueries.getIndex('temp')
        if getPineconeIndex:
            PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
            vectorStore = PineconeVectorStore(index_name='temp',embedding=self.embeding,pinecone_api_key=PINECONE_API_KEY )
            if vectorStore:
                store = vectorStore.from_existing_index(index_name='temp',embedding=self.embeding)                
            return store
        else:
            return vectorStore
        
    
    def getChunkedDocsFromPDF(self):
        currentDir = os.path.dirname(__file__)
        pdfPath = os.path.join(currentDir, '..', 'Airvon.pdf')
        loader = PyPDFLoader(pdfPath)
        pages = loader.load_and_split()
        textSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunkedDocs = textSplitter.split_documents(pages)        
        return chunkedDocs
        
        
        
    
    
    
        