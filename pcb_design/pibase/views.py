from rest_framework import generics, permissions,status
from rest_framework.views import APIView
from rest_framework.response import Response
from authentication.custom_permissions import IsAuthorized
from authentication.custom_authentication import CustomJWTAuthentication
from django.http import Http404
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .models import PiBaseComponent, PiBaseFieldCategory, PiBaseRecord, PiBaseFieldOption
from .serializers import (
    PiBaseComponentSerializer,
    PiBaseFieldCategorySerializer,
    PiBaseRecordSerializer,
    PiBaseFieldOptionSerializer,
    PiBaseRecordStepOneSerializer,
    BasicInfoSerializer,
    GeneralDetailsSerializer,
    ComponentsSelectionSerializer,
    PcbCanSerializer,
    CapacitorsSerializer,
    InductorsAircoilsTransformersSerializer,
    ResonatorSerializer,
    FinalComponentsSerializer,
    PreviewSerializer,
    PiBaseRecordGetSerializer,
    PiBaseRecordUniquenessSerializer,
    PiBaseRecordFullSerializer
)
import uuid

from masters.models import MstCategory,MstSubCategory

from . import pi_base_logs

# =====================================================================================================================================

class PiBaseComponentListView(generics.ListAPIView):
    queryset = PiBaseComponent.objects.all()
    serializer_class = PiBaseComponentSerializer
    pagination_class = None

# =====================================================================================================================================

class PiBaseFieldCategoryListView(generics.ListAPIView):
    queryset = PiBaseFieldCategory.objects.all()
    serializer_class = PiBaseFieldCategorySerializer
    pagination_class = None

# =====================================================================================================================================

class PiBaseFieldOptionListView(generics.ListAPIView):
    queryset = PiBaseFieldOption.objects.all()
    serializer_class = PiBaseFieldOptionSerializer
    pagination_class = None

# =====================================================================================================================================

from rest_framework.pagination import PageNumberPagination

class PiBaseRecordPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'  # optional, allow client to set page size in query param
    max_page_size = 100

# =====================================================================================================================================

class PiBaseRecordListView(generics.ListAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordSerializer
    pagination_class = PiBaseRecordPagination  # explicitly add pagination here

# =====================================================================================================================================

class PiBaseRecordStepOneCreateView(generics.CreateAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordStepOneSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

# =====================================================================================================================================

class GroupedFieldOptionsView(APIView):
    """
    API View to return grouped field options for PiBaseFieldCategory and selected MstCategory values.

    Returns a dictionary where each key corresponds to a group of selectable options,
    formatted as LABEL_VALUE pairs for dropdowns or similar components.

    Example Response:
    {
        "DIELECTRIC_MATERIAL_OPTIONS": [{"label": "FR4", "value": 1}, ...],
        "COPPER_THICKNESS_OPTIONS": [{"label": "35µm", "value": 3}, ...],
        ...
    }
    """

    def get(self, request):
        data = {}

        # Get PiBaseFieldCategory-related options
        try:
            categories = PiBaseFieldCategory.objects.filter(status=True).prefetch_related('options')
            for category in categories:
                key = category.name.upper().replace(' ', '_') + '_OPTIONS'
                options_list = [
                    {"label": option.value, "value": option.id}
                    for option in category.options.filter(status=True)
                ]
                data[key] = options_list
        except Exception as e:
            pi_base_logs.error(f"Error fetching PiBaseFieldCategory options: {str(e)}")
            return Response(
                {"detail": "Error retrieving field category options."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Custom MstCategory-based groups to include
        category_names_to_include = {
            "Dielectric material": "DIELECTRIC_MATERIAL_OPTIONS",
            "Copper Thickness": "COPPER_THICKNESS_OPTIONS",
            "Dielectric Thickness": "DIELECTRIC_THICKNESS_OPTIONS"
        }

        for cat_name, response_key in category_names_to_include.items():
            try:
                category = MstCategory.objects.get(category_name__iexact=cat_name)
                subcategories = MstSubCategory.objects.filter(category_Id=category.id)
                data[response_key] = [
                    {"label": sub.sub_category_name, "value": sub.id}
                    for sub in subcategories
                ]
            except MstCategory.DoesNotExist:
                pi_base_logs.warning(f"MstCategory not found for: {cat_name}")
                data[response_key] = []
            except Exception as e:
                pi_base_logs.error(f"Error fetching subcategories for {cat_name}: {str(e)}")
                data[response_key] = []

        return Response(data, status=status.HTTP_200_OK)
    
# =====================================================================================================================================

class CheckPiBaseRecordUniqueView(APIView):

    @swagger_auto_schema(request_body=PiBaseRecordUniquenessSerializer)
    def post(self, request, *args, **kwargs):
        serializer = PiBaseRecordUniquenessSerializer(data=request.data)
        if serializer.is_valid():
            op_no = serializer.validated_data.get('op_no')
            opu_no = serializer.validated_data.get('opu_no')
            edu_no = serializer.validated_data.get('edu_no')
            model_name = serializer.validated_data.get('model_name')

            results = {
                "op_no": {
                    "unique": not PiBaseRecord.objects.filter(op_no=op_no).exists(),
                },
                "opu_no": {
                    "unique": not PiBaseRecord.objects.filter(opu_no=opu_no).exists(),
                },
                "edu_no": {
                    "unique": not PiBaseRecord.objects.filter(edu_no=edu_no).exists(),
                },
                "model_name": {
                    "unique": not PiBaseRecord.objects.filter(model_name=model_name).exists(),
                }
            }

            # Add helpful message per field
            for field, result in results.items():
                result["message"] = (
                    f"{field} is unique." if result["unique"]
                    else f"{field} already exists."
                )

            return Response(results, status=status.HTTP_200_OK)

        # If input data itself is invalid
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# =====================================================================================================================================

class PiBaseRecordDetailAPIView(generics.CreateAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordFullSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

# =====================================================================================================================================
    
class PiBaseRecordDetailAPIViewUpdate(generics.RetrieveUpdateAPIView):
    serializer_class = PiBaseRecordFullSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get_queryset(self):
        return PiBaseRecord.objects.all()

    def get_object(self):
        """
        Match UUID generated from model ID using uuid5(PiBase-{id}).
        """
        record_uuid = self.kwargs.get('record_id')
        for obj in self.get_queryset():
            generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}')
            if str(generated_uuid) == str(record_uuid):
                return obj
        raise Http404("Record not found")

# =====================================================================================================================================

class PiBaseRecordPartialUpdateView(generics.RetrieveUpdateAPIView):
    queryset = PiBaseRecord.objects.all()
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]
    lookup_field = 'id'

    def get_serializer_class(self):
        """
        Dynamically choose the serializer based on currentStep from request.
        Fallback to BasicInfoSerializer if not found or for schema generation.
        """
        step = self.request.data.get('currentStep', 1) if self.request.method != 'GET' else 1
        serializer_map = {
            1: BasicInfoSerializer,
            2: GeneralDetailsSerializer,
            3: ComponentsSelectionSerializer,
            4: PcbCanSerializer,
            5: CapacitorsSerializer,
            6: InductorsAircoilsTransformersSerializer,
            7: ResonatorSerializer,
            8: FinalComponentsSerializer,
            9: PreviewSerializer,  # Assuming you have a PreviewSerializer for step 9
        }
        return serializer_map.get(int(step), BasicInfoSerializer)

    def get_object(self):
        """
        Match UUID generated from model id.
        """
        record_uuid = self.kwargs.get('record_id')
        for obj in self.get_queryset():
            generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}')
            if str(generated_uuid) == str(record_uuid):
                return obj
        raise Http404("Record not found")

    @swagger_auto_schema(
    operation_description="Partially update a PiBase record using a UUID and dynamic form step (1 to 8).",
    manual_parameters=[
        openapi.Parameter(
            'record_id',
            openapi.IN_PATH,
            description="UUID generated from model ID using uuid5 with NAMESPACE_DNS",
            type=openapi.TYPE_STRING,
            required=True
        )
    ],
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=["currentStep"],
        properties={
            'currentStep': openapi.Schema(type=openapi.TYPE_INTEGER, description="Step number (1 to 8)"),

            # --- Sample Fields for Each Step ---
            'name': openapi.Schema(type=openapi.TYPE_STRING, description="(Step 1) Basic Info - name"),
            'age': openapi.Schema(type=openapi.TYPE_INTEGER, description="(Step 1) Basic Info - age"),

            'location': openapi.Schema(type=openapi.TYPE_STRING, description="(Step 2) General Details - location"),

            'component_ids': openapi.Schema(type=openapi.TYPE_ARRAY,
                                            items=openapi.Items(type=openapi.TYPE_INTEGER),
                                            description="(Step 3) Components Selection - List of component IDs"),

            'pcb_type': openapi.Schema(type=openapi.TYPE_STRING, description="(Step 4) PCB Type"),

            'capacitor_value': openapi.Schema(type=openapi.TYPE_STRING, description="(Step 5) Capacitor Value"),

            'inductor_type': openapi.Schema(type=openapi.TYPE_STRING, description="(Step 6) Inductor Type"),

            'resonator_frequency': openapi.Schema(type=openapi.TYPE_STRING, description="(Step 7) Resonator Frequency"),

            'final_review_notes': openapi.Schema(type=openapi.TYPE_STRING, description="(Step 8) Final Notes"),
        }
    ),
    responses={200: "Record updated successfully", 400: "Validation error"}
)
    
    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', True)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data, status=status.HTTP_200_OK)


    def get(self, request, *args, **kwargs):
        instance = self.get_object() # This method should be defined in your actual view
        serializer = PiBaseRecordGetSerializer(instance)
        data = serializer.data

        transformed = {
            "opNumber": data.get("op_no"),
            "opuNumber": data.get("opu_no"),
            "eduNumber": data.get("edu_no"),
            "modelName": data.get("model_name"),
            "modelFamily": data.get("model_family"),
            "technology": data.get("technology"),
            "revisionNumber": data.get("revision_number"),
            "schematicFile": data.get("schematic"),
            "similarModel": data.get("similar_model_layout"),
            "impedance": data.get("impedance_selection", {}).get("impedance"),
            "customImpedance": data.get("impedance_selection", {}).get("customImpedance"),
            "bottomSolderMask": data.get("bottomSolderMask"),
            "halfMoonRequirement": data.get("halfMoonRequirement"),
            "viaHolesRequirement": data.get("viaHolesRequirement"),
            "signalLaunchType": data.get("signalLaunchType"),
            "coverType": data.get("coverType"),
            "designRuleViolation": data.get("designRuleViolation"),
            "currentStep": data.get("current_step"),
            "caseStyleType": data.get("case_style_data", {}).get("caseStyleType"),
            "caseStyle": data.get("case_style_data", {}).get("caseStyle"),
            "interfaces": data.get("package_details", {}).get("interfaces"),
            "caseDimensions": data.get("case_style_data", {}).get("caseDimensions"),
            "ports": data.get("package_details", {}).get("ports"),
            "enclosureDetails": data.get("package_details", {}).get("enclosureDetails"),
            "topcoverDetails": data.get("package_details", {}).get("topcoverDetails"),
            "can": data.get("can_details"),
            "selectedComponents": data.get("components", []),
            "capacitors": data.get("capacitor_details", {}),
            "inductors": data.get("inductor_details", {}),
            "airCoils": data.get("aircoil_details", {}),
            "resistors": data.get("resistor_details", {}),
            "transformers": data.get("transformer_details", {}),
            "pcbList": data.get("pcb_details", {}),
            "shieldList": data.get("shield_details", {}),
            "fingerList": data.get("finger_details", {}),
            "copperFlapList": data.get("copper_flaps_details", {}),
            "resonatorList": data.get("resonator_details", {}),
            "ltccList": data.get("ltcc_details", {}),
            "specialRequirements": data.get("special_requirements"),
        }

        return Response(transformed, status=status.HTTP_200_OK)

# =====================================================================================================================================
