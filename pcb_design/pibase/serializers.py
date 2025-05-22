# pibase/serializers.py

from rest_framework import serializers
from .models import PiBaseComponent,PiBaseFieldCategory,PiBaseRecord,PiBaseFieldOption

class PiBaseComponentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PiBaseComponent
        fields = '__all__'


class PiBaseFieldCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PiBaseFieldCategory
        fields = '__all__'


class PiBaseFieldOptionSerializer(serializers.ModelSerializer):
    category = PiBaseFieldCategorySerializer(read_only=True)  # nested category
    class Meta:
        model = PiBaseFieldOption
        fields = '__all__'

class PiBaseRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = PiBaseRecord
        fields = '__all__'
        