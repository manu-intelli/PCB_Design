# pibase/serializers.py
import uuid
from rest_framework import serializers
from .models import PiBaseComponent,PiBaseFieldCategory,PiBaseRecord,PiBaseFieldOption,PiBaseStatus
from django.contrib.auth import get_user_model


User = get_user_model()


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


class PiBaseRecordStepOneSerializer(serializers.ModelSerializer):
    recordId = serializers.SerializerMethodField()  # Add this field to serializer output

    opNumber = serializers.CharField(source='op_no')
    opuNumber = serializers.CharField(source='opu_no')
    eduNumber = serializers.CharField(source='edu_no')
    modelName = serializers.CharField(source='model_name')
    technology = serializers.PrimaryKeyRelatedField(queryset=PiBaseFieldOption.objects.all())
    modelFamily = serializers.PrimaryKeyRelatedField(queryset=PiBaseFieldOption.objects.all(), source='model_family')

    class Meta:
        model = PiBaseRecord
        fields = [
            'recordId',  # Include this in the output fields
            'opNumber', 'opuNumber', 'eduNumber', 'modelName',
            'technology', 'modelFamily'
        ]
        extra_kwargs = {
            'technology': {'required': True, 'allow_null': False},
            'model_family': {'required': True, 'allow_null': False}
        }

    def get_recordId(self, obj):
        # Generate a UUID based on the record's primary key (id)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}'))

    def validate(self, data):
        errors = {}
        if PiBaseRecord.objects.filter(op_no=data['op_no']).exists():
            errors['opNumber'] = 'OP Number already exists.'
        if PiBaseRecord.objects.filter(opu_no=data['opu_no']).exists():
            errors['opuNumber'] = 'OPU Number already exists.'
        if PiBaseRecord.objects.filter(edu_no=data['edu_no']).exists():
            errors['eduNumber'] = 'EDU Number already exists.'
        if PiBaseRecord.objects.filter(model_name=data['model_name']).exists():
            errors['modelName'] = 'Model Name already exists.'
        if errors:
            raise serializers.ValidationError(errors)
        return data

    def create(self, validated_data):
        request = self.context.get('request')
        user = request.user if request else None

        validated_data['status'] = PiBaseStatus.objects.get(pk=1)
        validated_data['current_step'] = 1
        validated_data['created_by'] = user
        validated_data['updated_by'] = user

        return super().create(validated_data)