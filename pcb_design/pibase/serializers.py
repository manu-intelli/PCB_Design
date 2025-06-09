# pibase/serializers.py
import uuid
from rest_framework import serializers
from .models import PiBaseComponent,PiBaseFieldCategory,PiBaseRecord,PiBaseFieldOption,PiBaseStatus,PiBaseImage
from django.contrib.auth import get_user_model

import json
import string
import random


User = get_user_model()



class StringifiedJSONField(serializers.JSONField):
    def to_internal_value(self, data):
        # If data is a string, try to parse JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise serializers.ValidationError(f"Invalid JSON string: {e}")
        return super().to_internal_value(data)
    
def safe_json_load(raw):
    try:
        return json.loads(raw) if raw else None
    except Exception:
        return raw  # or None, depending on what you prefer


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
    recordId = serializers.SerializerMethodField()  # add this field here
    class Meta:
        model = PiBaseRecord
        fields = '__all__'

    def get_recordId(self, obj):
        # Generate a UUID based on the record's primary key (id)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}'))


class PiBaseRecordUniquenessSerializer(serializers.Serializer):
    opNumber = serializers.CharField(source='op_number')
    opuNumber = serializers.CharField(source='opu_number')
    eduNumber = serializers.CharField(source='edu_number')
    modelName = serializers.CharField(source='model_name')

class PiBaseImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = PiBaseImage
        fields = '__all__'

    def _generate_unique_cookies(self):
        chars = string.ascii_letters + string.digits
        while True:
            random_str = ''.join(random.choices(chars, k=22))
            if not PiBaseImage.objects.filter(cookies=random_str).exists():
                return random_str

    def create(self, validated_data):
        if 'cookies' not in validated_data or not validated_data['cookies']:
            validated_data['cookies'] = self._generate_unique_cookies()
        return super().create(validated_data)

    def update(self, instance, validated_data):
        if 'cookies' not in validated_data or not validated_data['cookies']:
            validated_data['cookies'] = self._generate_unique_cookies()
        return super().update(instance, validated_data)



class BlankToNullIntegerField(serializers.IntegerField):
    def to_internal_value(self, data):
        if data == "":
            return None
        return super().to_internal_value(data)
    


class PiBaseRecordFullSerializer(serializers.ModelSerializer):
    # Field aliases
    opNumber = serializers.CharField(source='op_number')
    opuNumber = serializers.CharField(source='opu_number')
    eduNumber = serializers.CharField(source='edu_number')
    modelName = serializers.CharField(source='model_name')

    # Foreign key fields
    modelFamily = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.filter(category__name='Model Family'),
        source='model_family', required=False, allow_null=True
    )
    technology = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.all(), required=False, allow_null=True
    )
    status = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseStatus.objects.all(), default=1
    )

    # Input fields (raw)
    impedance = serializers.CharField(write_only=True)
    customImpedance = serializers.CharField(write_only=True, required=False, allow_blank=True)
    interfaces = serializers.CharField(write_only=True, required=False, allow_blank=True)
    caseStyleType = serializers.CharField(write_only=True, required=False, allow_blank=True)
    caseStyle = serializers.CharField(write_only=True, required=False, allow_blank=True)
    caseDimensions = serializers.JSONField(write_only=True, required=False)
    ports = serializers.JSONField(write_only=True, required=False)
    enclosureDetails = serializers.JSONField(write_only=True, required=False)
    topcoverDetails = serializers.JSONField(write_only=True, required=False)

    # Dropdowns
    bottomSolderMask = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.filter(category__name='Bottom Solder Mask'),
        source='bottom_solder_mask', required=False, allow_null=True
    )
    halfMoonRequirement = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.filter(category__name='Half Moon Requirement'),
        source='half_moon_requirement', required=False, allow_null=True
    )
    viaHolesRequirement = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.filter(category__name='Via Holes Requirement'),
        source='via_holes_requirement', required=False, allow_null=True
    )
    signalLaunchType = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.filter(category__name='Signal Launch Type'),
        source='signal_launch_type', required=False, allow_null=True
    )
    coverType = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.filter(category__name='Cover Type'),
        source='cover_type', required=False, allow_null=True
    )
    designRuleViolation = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.filter(category__name='Design Rule Violation'),
        source='design_rule_violation', required=False, allow_null=True
    )

    # Nested JSON fields
    impedanceSelection = serializers.JSONField(source='impedance_selection', required=False)
    interfacesDetails = serializers.JSONField(source='interfaces_details', required=False)
    caseStyleData = serializers.JSONField(source='case_style_data', required=False)
    canDetails = serializers.JSONField(source='can_details', required=False)
    pcbDetails = serializers.JSONField(source='pcb_details', required=False)
    chipAircoilDetails = serializers.JSONField(source='chip_aircoil_details', required=False)
    chipInductorDetails = serializers.JSONField(source='chip_inductor_details', required=False)
    chipCapacitorDetails = serializers.JSONField(source='chip_capacitor_details', required=False)
    chipResistorDetails = serializers.JSONField(source='chip_resistor_details', required=False)
    transformerDetails = serializers.JSONField(source='transformer_details', required=False)
    shieldDetails = serializers.JSONField(source='shield_details', required=False)
    fingerDetails = serializers.JSONField(source='finger_details', required=False)
    copperFlapsDetails = serializers.JSONField(source='copper_flaps_details', required=False)
    resonatorDetails = serializers.JSONField(source='resonator_details', required=False)
    ltccDetails = serializers.JSONField(source='ltcc_details', required=False)

    selectedComponents = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseComponent.objects.all(), source='components', many=True, required=False
    )

    schematicFile = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseImage.objects.all(), source='schematic', required=False, allow_null=True
    )

    schematicData = PiBaseImageSerializer(source='schematic', read_only=True)

    similarModel = serializers.CharField(source='similar_model_layout', required=False, allow_blank=True)
    specialRequirements = serializers.CharField(source='special_requirements', required=False, allow_blank=True)

    class Meta:
        model = PiBaseRecord
        fields = [
            'id', 'opNumber', 'opuNumber', 'eduNumber', 'modelName', 'modelFamily',
            'technology', 'status', 'revision_number',
            'impedance', 'customImpedance', 'interfaces', 'caseStyleType', 'caseStyle',
            'caseDimensions', 'ports', 'enclosureDetails', 'topcoverDetails',
            'bottomSolderMask', 'halfMoonRequirement', 'viaHolesRequirement',
            'signalLaunchType', 'coverType', 'designRuleViolation',
            'impedanceSelection', 'interfacesDetails', 'caseStyleData',
            'canDetails', 'pcbDetails', 'chipAircoilDetails', 'chipInductorDetails',
            'chipCapacitorDetails', 'chipResistorDetails', 'transformerDetails',
            'shieldDetails', 'fingerDetails', 'copperFlapsDetails', 'resonatorDetails',
            'ltccDetails', 'selectedComponents', 'schematicFile','schematicData',
            'similarModel', 'specialRequirements', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'revision_number' ,'created_at', 'updated_at']

    def validate(self, data):
        errors = {}
        instance = getattr(self, 'instance', None)
        
        # Get the values from the data or from the instance if updating
        op_number = data.get('op_number', instance.op_number if instance else None)
        opu_number = data.get('opu_number', instance.opu_number if instance else None)
        edu_number = data.get('edu_number', instance.edu_number if instance else None)
        model_name = data.get('model_name', instance.model_name if instance else None)
        
        # Check for duplicates, excluding the current instance if updating
        if op_number:
            qs = PiBaseRecord.objects.filter(op_number=op_number)
            if instance:
                qs = qs.exclude(pk=instance.pk)
            if qs.exists():
                errors['opNumber'] = 'OP Number already exists.'
        
        if opu_number:
            qs = PiBaseRecord.objects.filter(opu_number=opu_number)
            if instance:
                qs = qs.exclude(pk=instance.pk)
            if qs.exists():
                errors['opuNumber'] = 'OPU Number already exists.'
        
        if edu_number:
            qs = PiBaseRecord.objects.filter(edu_number=edu_number)
            if instance:
                qs = qs.exclude(pk=instance.pk)
            if qs.exists():
                errors['eduNumber'] = 'EDU Number already exists.'
        
        if model_name:
            qs = PiBaseRecord.objects.filter(model_name=model_name)
            if instance:
                qs = qs.exclude(pk=instance.pk)
            if qs.exists():
                errors['modelName'] = 'Model Name already exists.'
        
        if errors:
            raise serializers.ValidationError(errors)
        
        return data

    def assign_json_fields(self, instance):
        instance.impedance_selection = {
            "impedance": self.initial_data.get("impedance"),
            "customImpedance": self.initial_data.get("customImpedance")
        }
        instance.package_details = {
            "interfaces": self.initial_data.get("interfaces"),
            "ports": self.initial_data.get("ports"),
            "enclosureDetails": self.initial_data.get("enclosureDetails"),
            "topcoverDetails": self.initial_data.get("topcoverDetails")
        }
        instance.case_style_data = {
            "caseStyleType": self.initial_data.get("caseStyleType"),
            "caseStyle": self.initial_data.get("caseStyle"),
            "caseDimensions": self.initial_data.get("caseDimensions"),
        }

    def create(self, validated_data):
        user = self.context['request'].user
        components = validated_data.pop('components', [])
        
        # Pop write-only/non-model fields before model creation
        validated_data.pop("impedance", None)
        validated_data.pop("customImpedance", None)
        validated_data.pop("interfaces", None)
        validated_data.pop("caseStyleType", None)
        validated_data.pop("caseStyle", None)
        validated_data.pop("caseDimensions", None)
        validated_data.pop("ports", None)
        validated_data.pop("enclosureDetails", None)
        validated_data.pop("topcoverDetails", None)

        validated_data['created_by'] = user
        validated_data['updated_by'] = user
        validated_data['revision_number'] = "1"

        instance = PiBaseRecord.objects.create(**validated_data)

        # Now handle those popped fields into JSON fields
        self.assign_json_fields(instance)

        instance.save()

        if components:
            instance.components.set(components)

        return instance

    def update(self, instance, validated_data):
        user = self.context['request'].user
        components = validated_data.pop('components', None)

        # Update simple fields from validated_data
        for attr, value in validated_data.items():
            setattr(instance, attr, value)

        # Assign JSON fields from the original request data (self.initial_data)
        self.assign_json_fields(instance)

        # Update the 'updated_by' and increment 'revision_number'
        instance.updated_by = user
        try:
            instance.revision_number = str(int(instance.revision_number) + 1)
        except (ValueError, TypeError):
            instance.revision_number = "1"  # fallback if revision_number is invalid or None

        instance.save()

        # Update many-to-many relationship if components were passed
        if components is not None:
            instance.components.set(components)

        return instance
    
    