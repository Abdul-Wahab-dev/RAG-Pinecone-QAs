from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST

from lib.langchain import LangchainClient
from utils.perform_query import execute_raw_query
from django.http import JsonResponse
class RAGAPI(APIView):
    def post(self,request):
        files = request.FILES.getlist('files')
        client = LangchainClient()
        
        for file in files:
            docs = client.getChunkedDocsFromPDF(file)
            vectorStore =client.embedData(docs=docs)
            print(docs)
            
        return Response({'msg':'trigger'}, status=HTTP_200_OK)
    

class RAGQA(APIView):
    def __init__(self, **kwargs):
        self.langchainClient = LangchainClient()
    def post(self , request):
        data = request.data
        if data['query']:
            result = self.langchainClient.simpleFunctionCalling(query=f"{data['query']}")        
            return Response({'msg':'success', 'result':result},status=HTTP_200_OK)
        result = 'Please provide the query for search'        
        return Response({'msg':'success', 'result':result},status=HTTP_200_OK)



class DBRAG(APIView):
    def __init__(self, **kwargs):
        self.langChainClient = LangchainClient()
    def post(self, request):
        
        data = request.data
        if data['query']:
            result = self.langChainClient.qaDB(query=f"{data['query']}")        
            return Response({'msg':'success', 'result':result},status=HTTP_200_OK)
        result = 'Please provide the query for search'        
        return Response({'msg':'success', 'result':result},status=HTTP_200_OK)
        
    