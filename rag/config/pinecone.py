from pinecone import Pinecone
import os

class PineconeClient:
    
    def __init__(self):
        self.pc=Pinecone(api_key=os.environ.get('PINECONE_API_KEY')) 
        print(self.pc.list_indexes().names())
        if 'temp' not in self.pc.list_indexes().names():
            self.pc.create_index(name='temp', dimension=1536, metric='cosine')
        