import uuid
from rest_framework import serializers
from pibase.models import PiBaseRecord, PiBaseComponent


class PiBaseToMakeBillRecordSerializer(serializers.ModelSerializer):
    """
    Serializer for PiBaseRecord model with UUID-based recordId and user's full name.
    """
    recordId = serializers.SerializerMethodField()
    created_by_name = serializers.SerializerMethodField()

    class Meta:
        model = PiBaseRecord
        fields = [
            *[f.name for f in PiBaseRecord._meta.fields],
            'recordId',
            'created_by_name',
        ]

    def get_recordId(self, obj):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}'))

    def get_created_by_name(self, obj):
        user = getattr(obj, 'created_by', None)
        if user:
            full_name = user.get_full_name()
            return full_name or user.username or str(user)
        return ""

# Component name to model field name mapping
COMPONENT_TO_FIELD_MAP = {
    "PCB Name": "pcb_details",
    "CAN Details": "can_details",
    "Chip Capacitors": "chip_capacitor_details",
    "Chip Inductors": "chip_inductor_details",
    "Air Coil": "chip_aircoil_details",
    "Transformer": "transformer_details",
    "Shield": "shield_details",
    "Finger Details": "finger_details",
    "Copper Flap Details": "copper_flaps_details",
    "Resonator Details": "resonator_details",
    "Chip Resistor": "chip_resistor_details",
    "LTCC Details": "ltcc_details",
    # "Others" can be ignored or handled separately if needed
}

class PiBaseComponentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PiBaseComponent
        fields = "__all__"

class MakeBillRecordGetSerializer(serializers.ModelSerializer):
    created_by_name = serializers.SerializerMethodField()
    components = PiBaseComponentSerializer(many=True, read_only=True)

    class Meta:
        model = PiBaseRecord
        fields = [
            # Base fields
            'id', 'model_name', 'op_number', 'opu_number', 'edu_number',
            'revision_number', 'created_at', 'created_by', 'status',
            'created_by_name', 'components',
            'case_style_data', 'special_requirements',
            # All possible JSON fields (conditionally included)
            'can_details', 'pcb_details', 'chip_aircoil_details',
            'chip_inductor_details', 'chip_capacitor_details',
            'chip_resistor_details', 'transformer_details', 'shield_details',
            'finger_details', 'copper_flaps_details', 'resonator_details',
            'ltcc_details'
        ]

    def get_created_by_name(self, obj):
        user = getattr(obj, 'created_by', None)
        if user:
            full_name = user.get_full_name()
            return full_name or user.username or str(user)
        return ""

    def to_representation(self, instance):
        data = super().to_representation(instance)

        # Get selected component names
        selected_components = data.get("components", [])
        selected_field_keys = {
            COMPONENT_TO_FIELD_MAP.get(comp.get("name"))
            for comp in selected_components
            if COMPONENT_TO_FIELD_MAP.get(comp.get("name"))
        }

        # Fields that should always be returned
        base_fields = {
            'id', 'model_name', 'op_number', 'opu_number', 'edu_number',
            'revision_number', 'created_at', 'created_by', 'status',
            'created_by_name', 'components', 'case_style_data', 'special_requirements'
        }

        # Return only selected JSON fields + base fields
        return {
            key: value
            for key, value in data.items()
            if key in base_fields or key in selected_field_keys
        }