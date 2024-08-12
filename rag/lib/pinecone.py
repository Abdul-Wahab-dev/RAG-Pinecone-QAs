from config.pinecone import PineconeClient



class PineconeQueries:
    client = None
    def __init__(self):
        self.client = PineconeClient()
    def createIndex(self, name):
        self.client.pc.create_index(name=name,dimension=1536,metric='cosine')
    
    def initializePineconeClient(self,*args):
        index = args[0]
        if index not in self.client.pc.list_indexes().names():
            self.createIndex('temp')

    def getIndex(self,*args):
        name = args[0]
        index = self.client.pc.Index(name=name)
        return index