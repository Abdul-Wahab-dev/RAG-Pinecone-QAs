from rest_framework import serializers

class APISerializer(serializers.Serializer):
    name = serializers.CharField(max_length=200)