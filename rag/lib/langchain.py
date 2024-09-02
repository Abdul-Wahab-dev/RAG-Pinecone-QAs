from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,PromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_pinecone import PineconeVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from lib.pinecone import PineconeQueries
from pypdf import PdfReader
from openai import OpenAI
from langchain.docstore.document import Document
from langchain_core.messages import AIMessage, HumanMessage
from io import BytesIO
from langchain_community.vectorstores import Chroma

import os
import json
from langchain_pinecone import Pinecone
chat_history = []
class LangchainClient:
    def __init__(self):
        print(os.environ.get('OPENAI_KEY'),'os.environ.get()')
        self.embeding = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_KEY'), model='text-embedding-ada-002')
        self.openaiClient = ChatOpenAI(api_key=os.environ.get('OPENAI_KEY'), model='gpt-3.5-turbo-16k')
        self.pineconeQueries = PineconeQueries()
        self.aiClient = OpenAI(api_key=os.environ.get('OPENAI_KEY'))
    def qaDB(self, *args , **kwargs):
        db = SQLDatabase.from_uri(f"mysql+pymysql://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}@{os.environ.get('DB_HOST')}/{os.environ.get('DATABASE')}")
        print(db.dialect)
        print(db.get_usable_table_names())
        query = kwargs['query']
       
        examples = [
            {
                'input':"What is the id of the 'Pool' game?",
                'query':'SELECT id FROM games WHERE title = "Pool"'
            },
            {
                'input':" can you get the game who's id is 10",
                'query':'SELECT `id`, `status`, `thumbnail`, `title`, `configs`, `createdAt`, `updatedAt`, `gameStoreId`, `webUrl`, `zipUrl`, `matchMakingImage`, `version`, `streamId`, `playGameThumbnail`, `similarGamesThumbnail` FROM games WHERE `id` = 10'
            },
            {
                'input':"can you get the most game played by the user id is 1?",
                'query':'SELECT COUNT(*) AS totalGamesPlayed, userId FROM match_players GROUP BY userId HAVING userId = 1 ORDER BY totalGamesPlayed DESC LIMIT 1;'
            },
            {
                'input':"How many number of games user id 1 is win?",
                'query':'SELECT COUNT(*) AS total_wins FROM leaderboards WHERE userId = 1'
            },
        ]
        
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human","{input}\nSQLQuery:"),
                ("ai", "{query}" )
            ]
        )
        # getIndex = self.pineconeQueries.getIndex('temp')
        # vectore_store = Pinecone(pinecone_api_key=os.environ.get('PINECONE_API_KEY'),embedding=self.embeding,index=getIndex)
        vectore_store = Chroma()
        vectore_store.delete_collection()
        example_selector = SemanticSimilarityExampleSelector.from_examples(examples=examples, embeddings=self.embeding, vectorstore_cls=vectore_store, input_keys=['input'], k=2)
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, example_selector=example_selector, input_variables=['input','top_k'])
        
        final_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
                    few_shot_prompt,
                    ("human", "{input}"),
                ]
                )    
        generate_query = create_sql_query_chain(llm=self.openaiClient, db=db, prompt=final_prompt) 
        # query = generate_query.invoke({'question': 'Which game is played the most of time?'})
        execute_query = QuerySQLDataBaseTool(db=db)
        # result = execute_query.invoke(query)
        
        answer_prompt = PromptTemplate.from_template(
                """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
            )
        
        rephrase_answer = answer_prompt | self.openaiClient | StrOutputParser()
        chain = (
            RunnablePassthrough.assign(query=generate_query).assign(result=itemgetter('query') | execute_query)
        ) | rephrase_answer
        
        # print(few_shot_prompt.format(input='What is the id of Pool game'))
        result = chain.invoke({'question': query})
        
        return result
    def qaChain(self , *args , **kwargs):
        def get_candidate_info(name, email,skills,experience):

            # Example output returned from an API or database
            candidate_info = {
                "name": name,
                "email": email,
                "skills": skills,
                "experience": experience,
                }

            return json.dumps(candidate_info)
        
        function_description = [
           {
               
           "type":"function",
           "function": {
            "name": "get_candidate_info",
            "description":"Get the candidate information about skills, experience , name and email",
            "parameters": {
                "type": "object",
                "properties":{
                    "name":{
                        "type":"string",
                        "description":"Candidate name, e.g. John Snow"
                    },
                    "email":{
                        "type":"string",
                        "description":"Candidate email, e.g. johnsnow@gmail.com"
                    },
                    "skills": {
                        "type":"array",
                        "description": 'List of candidate skills , e.g. ["Python", "SQL"]',
                        "items":{
                            "type": "string"
                        }
                        
                    },
                    "experience":{
                        "type":"array",
                        "description": "List of candidate experience in different company , e.g. ['Meta' , 'Ripeseed']",
                        "items":{
                            "type": "string"
                        }
                    }
                },
                'required': ['name','email', 'skills','experience']
                }
            }
            
           }
        ]
        
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
        
        system_prompt = """
                You are an assistant for short-listing the candidates tasks.
                Use the following pieces of retrieved context to answer 
                the question. 
                If you don't know the answer, say that you 
                don't know.keep the 
                answer concise.
                {context}
                """
            
        prompt = ChatPromptTemplate.from_messages(
                [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                ]
            )
        
        llm_with_function_call = self.openaiClient.bind_tools(tools=function_description)
        # llm_with_function_call = self.openaiClient
        question_answer_chain = create_stuff_documents_chain(llm=llm_with_function_call, prompt=prompt)
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
        
        return result
        
    def simpleFunctionCalling(self, *args, **kwargs):
        query = kwargs['query']
        def get_flight_info(loc_origin, loc_destination):
            # Example output returned from an API or database
            flight_info = {
                "loc_origin": loc_origin,
                "loc_destination": loc_destination,
            }

            return json.dumps(flight_info)
        function_description = [
    {
        "name": "get_flight_info",
        "description": "Get flight information between two locations",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
                    },
                    "required": ["loc_origin", "loc_destination"],
                },
            }
        ]
        completion = self.aiClient.chat.completions.create(model='gpt-3.5-turbo-16k', 
                                                           messages=[{'role': 'user', "content": query}],
                                                           functions=function_description,
                                                           function_call='auto'
                                                           )
        result = completion.choices[0].message
        origin = json.loads(result.function_call.arguments).get("loc_origin")
        destination = json.loads(result.function_call.arguments).get("loc_destination")
        params = json.loads(result.function_call.arguments)
        chosen_function = eval(result.function_call.name)
        flight = chosen_function(**params)
        return flight
        
    def embedData(self,*args , **kwargs):
        data = [docs for docs in kwargs['docs']]
        
        
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
            return store.as_retriever(search_type="similarity", search_kwargs={'k': 9})
        else:
            return vectorStore
        
    
    def getChunkedDocsFromPDF(self, file):
        name = file.name
        print(name)
        # load file in memory 
        pdf_file = BytesIO(file.read())
        
        pdf_reader = PdfReader(pdf_file)
        
        documents = []
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            documents.append(Document(page_content=text,metadata={"source": name}))
        # currentDir = os.path.dirname(__file__)
        # pdfPath = os.path.join(currentDir, '..', 'Airvon.pdf')
        # loader = PyPDFLoader(pdf_file)
        # pages = loader.load_and_split()
        # textSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        # chunkedDocs = textSplitter.split_documents(pdf_reader.pages)        
        return documents
        
        
        
    
    
    
        