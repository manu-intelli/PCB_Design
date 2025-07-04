from rest_framework import serializers
from .models import FilterSubmission

class FilterSubmissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = FilterSubmission
        fields = '__all__'
