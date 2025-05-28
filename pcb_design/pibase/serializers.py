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
    recordId = serializers.SerializerMethodField()  # add this field here
    class Meta:
        model = PiBaseRecord
        fields = '__all__'

    def get_recordId(self, obj):
        # Generate a UUID based on the record's primary key (id)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}'))

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


class PiBaseRecordGetSerializer(serializers.ModelSerializer):
    # ForeignKey fields to their .name
    technology = serializers.CharField(source='technology.name', default=None)
    model_family = serializers.CharField(source='model_family.name', default=None)
    bottom_solder_mask = serializers.CharField(source='bottom_solder_mask.name', default=None)
    half_moon_requirement = serializers.CharField(source='half_moon_requirement.name', default=None)
    via_holes_on_signal_pads = serializers.CharField(source='via_holes_on_signal_pads.name', default=None)
    signal_launch_type = serializers.CharField(source='signal_launch_type.name', default=None)
    cover_type = serializers.CharField(source='cover_type.name', default=None)
    design_rule_violation_accepted = serializers.CharField(source='design_rule_violation_accepted.name', default=None)

    # ManyToMany components as list of component names
    components = serializers.SlugRelatedField(many=True, read_only=True, slug_field='name')

    class Meta:
        model = PiBaseRecord
        fields = [
            'op_no', 'opu_no', 'edu_no', 'model_name', 'schematic', 'similar_model_layout', 'revision_number',
            'technology', 'model_family', 'bottom_solder_mask', 'half_moon_requirement', 'via_holes_on_signal_pads',
            'signal_launch_type', 'cover_type', 'design_rule_violation_accepted',
            'impedance_selection', 'package_details', 'case_style_data',
            'components',
            'can_details', 'pcb_details', 'aircoil_details', 'inductor_details', 'capacitor_details', 'resistor_details',
            'transformer_details', 'shield_details', 'finger_details', 'copper_flaps_details', 'resonator_details', 'ltcc_details',
            'status', 'created_by', 'updated_by', 'current_step',
            'created_at', 'updated_at',
        ]

    # Override to_representation to convert snake_case to camelCase in output keys
    def to_representation(self, instance):
        data = super().to_representation(instance)
        
        def snake_to_camel(snake_str):
            parts = snake_str.split('_')
            return parts[0] + ''.join(word.capitalize() for word in parts[1:])
        
        return {snake_to_camel(k): v for k, v in data.items()}

class BasicInfoSerializer(serializers.ModelSerializer):
    opNumber = serializers.CharField(source='op_no')
    opuNumber = serializers.CharField(source='opu_no')
    eduNumber = serializers.CharField(source='edu_no')
    modelName = serializers.CharField(source='model_name')
    currentStep = serializers.IntegerField(source='current_step')
    technology = serializers.PrimaryKeyRelatedField(queryset=PiBaseFieldOption.objects.all())
    modelFamily = serializers.PrimaryKeyRelatedField(
        queryset=PiBaseFieldOption.objects.all(), 
        source='model_family'
    )

    class Meta:
        model = PiBaseRecord
        fields = [
            'opNumber', 'opuNumber', 'eduNumber', 'modelFamily', 'modelName',
            'technology', 'revision_number', 'currentStep'
        ]
        extra_kwargs = {
            'modelFamily': {'required': True},
            'technology': {'required': True},
        }

    def validate(self, data):
        # Get instance if present (in PATCH)
        instance = self.instance
        errors = {}

        # Extract values, fallback to instance if not in PATCH payload
        op_no = data.get('op_no', getattr(instance, 'op_no', None))
        opu_no = data.get('opu_no', getattr(instance, 'opu_no', None))
        edu_no = data.get('edu_no', getattr(instance, 'edu_no', None))
        model_name = data.get('model_name', getattr(instance, 'model_name', None))

        if PiBaseRecord.objects.filter(op_no=op_no).exclude(id=instance.id).exists():
            errors['opNumber'] = 'OP Number already exists.'
        if PiBaseRecord.objects.filter(opu_no=opu_no).exclude(id=instance.id).exists():
            errors['opuNumber'] = 'OPU Number already exists.'
        if PiBaseRecord.objects.filter(edu_no=edu_no).exclude(id=instance.id).exists():
            errors['eduNumber'] = 'EDU Number already exists.'
        if PiBaseRecord.objects.filter(model_name=model_name).exclude(id=instance.id).exists():
            errors['modelName'] = 'Model Name already exists.'

        if errors:
            raise serializers.ValidationError(errors)
        return data

    def update(self, instance, validated_data):
        # Handle the current_step separately
        current_step = validated_data.pop('current_step', None)
        if current_step is not None:
            instance.current_step = current_step

        # Update remaining fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)

        instance.save()
        return instance


class GeneralDetailsSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    impedance = serializers.CharField(write_only=True)
    customImpedance = serializers.CharField(write_only=True, required=False,allow_null=True, allow_blank=True)
    interfaces = serializers.CharField(write_only=True)
    caseStyleType = serializers.CharField(write_only=True)
    CaseStyle = serializers.CharField(write_only=True, required=False)
    caseDimensions = serializers.JSONField(write_only=True)
    ports = serializers.JSONField(write_only=True)
    enclosureDetails = serializers.JSONField(write_only=True)
    topcoverDetails = serializers.JSONField(write_only=True)
    bottomSolderMask = serializers.CharField(write_only=True)
    halfMoonRequirement = serializers.CharField(write_only=True)
    viaHolesRequirement = serializers.CharField(write_only=True)
    signalLaunchType = serializers.CharField(write_only=True)
    coverType = serializers.CharField(write_only=True)
    designRuleViolation = serializers.CharField(write_only=True)
    similarModel = serializers.CharField(write_only=True, required=False)
    schematicFile = serializers.CharField(write_only=True, required=False,allow_null=True, allow_blank=True)

    class Meta:
        model = PiBaseRecord
        fields = [
            'currentStep', 'impedance', 'customImpedance', 'interfaces', 
            'caseStyleType', 'CaseStyle', 'caseDimensions', 'ports',
            'enclosureDetails', 'topcoverDetails', 'bottomSolderMask',
            'halfMoonRequirement', 'viaHolesRequirement', 'signalLaunchType',
            'coverType', 'designRuleViolation', 'similarModel', 'schematicFile'
        ]

    def get_currentStep(self, obj):
        return obj.current_step  # Adjust according to your model field name

    def validate(self, data):
        # Add validation logic here
        return data

    def update(self, instance, validated_data):
        instance.impedance_selection = {
            'impedance': validated_data.get('impedance'),
            'customImpedance': validated_data.get('customImpedance', '')
        }
        instance.package_details = {
            'interfaces': validated_data.get('interfaces'),
            'caseStyleType': validated_data.get('caseStyleType'),
            'CaseStyle': validated_data.get('CaseStyle', ''),
            'caseDimensions': validated_data.get('caseDimensions'),
            'ports': validated_data.get('ports'),
            'enclosureDetails': validated_data.get('enclosureDetails'),
            'topcoverDetails': validated_data.get('topcoverDetails')
        }
        
        
        if 'schematicFile' in validated_data:
            instance.schematic = validated_data['schematicFile']
        
        instance.current_step = self.initial_data.get('currentStep', instance.current_step)
        instance.save()
        return instance


class ComponentsSelectionSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    selectedComponents = serializers.SerializerMethodField()
    # components = serializers.SerializerMethodField()

    class Meta:
        model = PiBaseRecord
        fields = ['currentStep', 'selectedComponents']

    def get_currentStep(self, obj):
        return obj.current_step


    def get_selectedComponents(self, obj):
        # Same as components, because both reflect selected component IDs
        return [component.id for component in obj.components.all()]

    def update(self, instance, validated_data):
        # Get data from initial_data because selectedComponents is write-only originally
        component_ids = self.initial_data.get('selectedComponents', [])
        components = PiBaseComponent.objects.filter(id__in=component_ids)
        instance.components.set(components)

        instance.current_step = self.initial_data.get('currentStep', instance.current_step)
        instance.save()
        return instance


class PcbCanSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    can = serializers.JSONField(write_only=True)
    pcbList = serializers.JSONField(write_only=True)

    class Meta:
        model = PiBaseRecord
        fields = ['currentStep', 'can', 'pcbList']

    def get_currentStep(self, obj):
        return obj.current_step

    def update(self, instance, validated_data):
        instance.can_details = validated_data.get('can', {})
        instance.pcb_details = validated_data.get('pcbList', [])
        instance.current_step = self.initial_data.get('currentStep', instance.current_step)
        instance.save()
        return instance

class CapacitorsSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    capacitors = serializers.JSONField(write_only=True)

    class Meta:
        model = PiBaseRecord
        fields = ['currentStep', 'capacitors']

    def get_currentStep(self, obj):
        return obj.current_step

    def update(self, instance, validated_data):
        instance.capacitor_details = validated_data.get('capacitors', {})
        instance.current_step = self.initial_data.get('currentStep', instance.current_step)
        instance.save()
        return instance

class InductorsAircoilsTransformersSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    inductors = serializers.JSONField(write_only=True)
    airCoils = serializers.JSONField(write_only=True)
    transformers = serializers.JSONField(write_only=True)

    class Meta:
        model = PiBaseRecord
        fields = ['currentStep', 'inductors', 'airCoils', 'transformers']
    
    def get_currentStep(self, obj):
        return obj.current_step

    def update(self, instance, validated_data):
        instance.inductor_details = validated_data.get('inductors', {})
        instance.aircoil_details = validated_data.get('airCoils', {})
        instance.transformer_details = validated_data.get('transformers', {})
        instance.current_step = self.initial_data.get('currentStep', instance.current_step)
        instance.save()
        return instance

class ResonatorSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    resonatorList = serializers.JSONField(write_only=True)

    class Meta:
        model = PiBaseRecord
        fields = ['currentStep', 'resonatorList']

    def get_currentStep(self, obj):
        return obj.current_step

    def update(self, instance, validated_data):
        instance.resonator_details = validated_data.get('resonatorList', {})
        instance.current_step = self.initial_data.get('currentStep', instance.current_step)
        instance.save()
        return instance

class FinalComponentsSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    copperFlapList = serializers.JSONField(write_only=True)
    ltccList = serializers.JSONField(write_only=True)
    resistors = serializers.JSONField(write_only=True)
    shieldList = serializers.JSONField(write_only=True)
    fingerList = serializers.JSONField(write_only=True)
    specialRequirements = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = PiBaseRecord
        fields = [
            'currentStep', 'copperFlapList', 'ltccList', 'resistors',
            'shieldList', 'fingerList', 'specialRequirements'
        ]

    def get_currentStep(self, obj):
        return obj.current_step

    def update(self, instance, validated_data):
        instance.copper_flaps_details = validated_data.get('copperFlapList', {})
        instance.ltcc_details = validated_data.get('ltccList', {})
        instance.resistor_details = validated_data.get('resistors', {})
        instance.shield_details = validated_data.get('shieldList', {})
        instance.finger_details = validated_data.get('fingerList', {})
        
        if 'specialRequirements' in validated_data:
            instance.special_requirements = validated_data['specialRequirements']
        
        instance.current_step = self.initial_data.get('currentStep', instance.current_step)
        instance.save()
        return instance
    
