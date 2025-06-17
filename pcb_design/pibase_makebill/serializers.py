import uuid
from rest_framework import serializers
from pibase.models import PiBaseRecord

class PiBaseToMakeBillRecordSerializer(serializers.ModelSerializer):
    """
    Serializer for PiBaseRecord model, with a UUID-based recordId field and
    created_by_name as the user's full name.
    """
    recordId = serializers.SerializerMethodField()
    created_by_name = serializers.SerializerMethodField()

    class Meta:
        model = PiBaseRecord
        fields = [  # Explicitly list all fields you want to include
            *[f.name for f in PiBaseRecord._meta.fields],  # All model fields
            'recordId',
            'created_by_name',
        ]

    def get_recordId(self, obj):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}'))

    def get_created_by_name(self, obj):
        user = getattr(obj, 'created_by', None)
        if user:
            full_name = user.get_full_name()
            if full_name:
                return full_name
            return user.username or str(user)
        return ""