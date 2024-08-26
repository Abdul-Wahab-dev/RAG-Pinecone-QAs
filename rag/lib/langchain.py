from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain 
from langchain_pinecone import PineconeVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from lib.pinecone import PineconeQueries
from langchain_core.messages import AIMessage, HumanMessage
from config.pinecone import PineconeClient

import os
chat_history = []
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
        contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system" , contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        prompt = ChatPromptTemplate.from_messages(
                [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                ]
            )
        question_answer_chain = create_stuff_documents_chain(llm=self.openaiClient, prompt=prompt)
        retriever = self.getVectorStore()
        history_aware_retriever = create_history_aware_retriever(self.openaiClient,retriever,contextualize_q_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        # chain = VectorDBQA.from_chain_type(llm=self.openaiClient, chain_type='stuff', vectorstore=vector_store)
        result = rag_chain.invoke({'input': query, 'chat_history': chat_history})
        
        chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(content=result['answer'])
            ]
        )
        
        return result['answer']
        
        
        
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
            return store.as_retriever()
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
        
        
        
    
    
    
        