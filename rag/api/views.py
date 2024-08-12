from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST

from lib.langchain import LangchainClient


class RAGAPI(APIView):
    def post(self,request):
        client = LangchainClient()
        docs = client.getChunkedDocsFromPDF()
        vectorStore =client.embedData(docs=docs)
        return Response({'msg':'trigger', 'vector_store':vectorStore}, status=HTTP_200_OK)
    

class RAGQA(APIView):
    def __init__(self, **kwargs):
        self.langchainClient = LangchainClient()
    def post(self , request):
        data = request.data
        if data['query']:
            result = self.langchainClient.qaChain(query=f"{data['query']} you should only give the anwser from the provided context. If you do not find the answer then response with 'The provided question is out of context or not mention in the context anywhere'")        
            return Response({'msg':'success', 'result':result},status=HTTP_200_OK)
        result = 'Please provide the query for search'        
        return Response({'msg':'success', 'result':result},status=HTTP_200_OK)
        
    