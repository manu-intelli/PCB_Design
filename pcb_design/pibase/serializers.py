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
    # These SerializerMethodFields are correctly defined to use methods for retrieval
    technology = serializers.SerializerMethodField()
    model_family = serializers.SerializerMethodField()
    bottomSolderMask = serializers.SerializerMethodField()
    halfMoonRequirement = serializers.SerializerMethodField()
    viaHolesRequirement = serializers.SerializerMethodField()
    signalLaunchType = serializers.SerializerMethodField()
    coverType = serializers.SerializerMethodField()
    designRuleViolation = serializers.SerializerMethodField()
    components = serializers.SerializerMethodField() # Use read_only=True for GET operations

    class Meta:
        model = PiBaseRecord
        fields = [
            'op_no', 'opu_no', 'edu_no', 'model_name', 'model_family', 'technology',
            'revision_number', 'schematic', 'similar_model_layout',
            'impedance_selection', 'bottomSolderMask', 'halfMoonRequirement',
            'viaHolesRequirement', 'signalLaunchType', 'coverType', 'designRuleViolation',
            'current_step', 'package_details', 'case_style_data','components',
            'capacitor_details', 'inductor_details', 'aircoil_details',
            'resistor_details', 'transformer_details', 'can_details',
            'pcb_details', 'shield_details', 'finger_details',
            'copper_flaps_details', 'resonator_details', 'ltcc_details','special_requirements',
        ]

    # Corrected methods to access .value instead of .name
    def get_technology(self, obj):
        return obj.technology.id if obj.technology else None

    def get_model_family(self, obj):
        return obj.model_family.id if obj.model_family else None

    def get_bottomSolderMask(self, obj):
        return obj.bottom_solder_mask.id if obj.bottom_solder_mask else None

    def get_halfMoonRequirement(self, obj):
        return obj.half_moon_requirement.id if obj.half_moon_requirement else None

    def get_viaHolesRequirement(self, obj):
        return obj.via_holes_on_signal_pads.id if obj.via_holes_on_signal_pads else None

    def get_signalLaunchType(self, obj):
        return obj.signal_launch_type.id if obj.signal_launch_type else None

    def get_coverType(self, obj):
        return obj.cover_type.id if obj.cover_type else None

    def get_designRuleViolation(self, obj):
        return obj.design_rule_violation_accepted.id if obj.design_rule_violation_accepted else None
    
    def get_components(self, obj):
        # obj.components is a ManyToManyManager. .all() retrieves the related objects.
        # Then, a list comprehension extracts the 'id' of each component.
        return [component.id for component in obj.components.all()]

class BasicInfoSerializer(serializers.ModelSerializer):
    opNumber = serializers.CharField(source='op_no')
    opuNumber = serializers.CharField(source='opu_no')
    eduNumber = serializers.CharField(source='edu_no')
    modelName = serializers.CharField(source='model_name')
    currentStep = serializers.IntegerField(source='current_step')
    technology = serializers.IntegerField()
    modelFamily = serializers.IntegerField(source='model_family')

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


class BlankToNullIntegerField(serializers.IntegerField):
    def to_internal_value(self, data):
        if data == "":
            return None
        return super().to_internal_value(data)
    

class GeneralDetailsSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    impedance = serializers.CharField(write_only=True)
    customImpedance = serializers.CharField(write_only=True, required=False, allow_null=True, allow_blank=True)
    interfaces = serializers.CharField(write_only=True)
    caseStyleType = serializers.CharField(write_only=True)
    CaseStyle = serializers.CharField(write_only=True, required=False)
    caseDimensions = serializers.JSONField(write_only=True)
    ports = serializers.JSONField(write_only=True)
    enclosureDetails = serializers.JSONField(write_only=True)
    topcoverDetails = serializers.JSONField(write_only=True)

    bottomSolderMask = BlankToNullIntegerField(write_only=True, allow_null=True)  # ✅ Now works!
    halfMoonRequirement = BlankToNullIntegerField(write_only=True, allow_null=True)
    viaHolesRequirement = BlankToNullIntegerField(write_only=True, allow_null=True)
    signalLaunchType = BlankToNullIntegerField(write_only=True, allow_null=True)
    coverType = BlankToNullIntegerField(write_only=True, allow_null=True)
    designRuleViolation = BlankToNullIntegerField(write_only=True, allow_null=True)


    similarModel = serializers.CharField(write_only=True, required=False)
    schematicFile = serializers.FileField(required=False, allow_null=True)

    class Meta:
        model = PiBaseRecord
        fields = [
            'currentStep', 'impedance', 'customImpedance', 'interfaces',
            'caseStyleType', 'CaseStyle', 'caseDimensions', 'ports',
            'enclosureDetails', 'topcoverDetails',
            'bottomSolderMask', 'halfMoonRequirement', 'viaHolesRequirement',
            'signalLaunchType', 'coverType', 'designRuleViolation',
            'similarModel', 'schematicFile'
        ]

    def get_currentStep(self, obj):
        return obj.current_step

    def validate(self, data):
        # Add any custom validation if required
        return data

    def update(self, instance, validated_data):
        # Handle simple JSON assignments
        instance.impedance_selection = {
            'impedance': validated_data.get('impedance'),
            'customImpedance': validated_data.get('customImpedance', '')
        }

        instance.package_details = {
            'interfaces': validated_data.get('interfaces'),
            
            'ports': validated_data.get('ports'),
            'enclosureDetails': validated_data.get('enclosureDetails'),
            'topcoverDetails': validated_data.get('topcoverDetails')
        }

        instance.case_style_data = {
            'caseStyleType': validated_data.get('caseStyleType'),
            'CaseStyle': validated_data.get('CaseStyle', ''),
            'caseDimensions': validated_data.get('caseDimensions'),
        }

        # Foreign key assignments with validation
        fk_fields = [
            ('bottom_solder_mask', 'bottomSolderMask'),
            ('half_moon_requirement', 'halfMoonRequirement'),
            ('via_holes_on_signal_pads', 'viaHolesRequirement'),
            ('signal_launch_type', 'signalLaunchType'),
            ('cover_type', 'coverType'),
            ('design_rule_violation_accepted', 'designRuleViolation'),
        ]

        for model_field, input_field in fk_fields:
            field_value = validated_data.get(input_field, None)
            if field_value is not None:
                try:
                    setattr(
                        instance,
                        model_field,
                        PiBaseFieldOption.objects.get(id=field_value)
                    )
                except PiBaseFieldOption.DoesNotExist:
                    raise serializers.ValidationError({input_field: "Invalid option ID."})
            else:
                setattr(instance, model_field, None)  # Allow null if not provided


        if 'schematicFile' in validated_data:
            instance.schematic = validated_data['schematicFile']
        if 'similarModel' in validated_data:
            instance.similar_model_layout = validated_data['similarModel']

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
    
class PreviewSerializer(serializers.ModelSerializer):
    currentStep = serializers.SerializerMethodField()
    status = BlankToNullIntegerField(write_only=True, allow_null=True)

    class Meta:
        model = PiBaseRecord
        fields = ['currentStep', 'status']

    def get_currentStep(self, obj):
        return obj.current_step

    def update(self, instance, validated_data):
        # Update status if provided
        if 'status' in validated_data:
            try:
                instance.status = PiBaseStatus.objects.get(id=validated_data['status'])
            except PiBaseStatus.DoesNotExist:
                raise serializers.ValidationError({'status': 'Invalid option ID.'})

        # Set current step
        instance.current_step = self.initial_data.get('currentStep', instance.current_step)

        instance.save()
        return instance

