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
    PiBaseRecordGetSerializer
)
import uuid

class PiBaseComponentListView(generics.ListAPIView):
    queryset = PiBaseComponent.objects.all()
    serializer_class = PiBaseComponentSerializer
    pagination_class = None

class PiBaseFieldCategoryListView(generics.ListAPIView):
    queryset = PiBaseFieldCategory.objects.all()
    serializer_class = PiBaseFieldCategorySerializer
    pagination_class = None

class PiBaseFieldOptionListView(generics.ListAPIView):
    queryset = PiBaseFieldOption.objects.all()
    serializer_class = PiBaseFieldOptionSerializer
    pagination_class = None

from rest_framework.pagination import PageNumberPagination

class PiBaseRecordPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'  # optional, allow client to set page size in query param
    max_page_size = 100

class PiBaseRecordListView(generics.ListAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordSerializer
    pagination_class = PiBaseRecordPagination  # explicitly add pagination here


class PiBaseRecordStepOneCreateView(generics.CreateAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordStepOneSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class GroupedFieldOptionsView(APIView):
    def get(self, request):
        categories = PiBaseFieldCategory.objects.filter(status=True).prefetch_related('options')
        data = {}

        for category in categories:
            # Convert category name to uppercase, replace spaces with underscores and append '_OPTIONS'
            key = category.name.upper().replace(' ', '_') + '_OPTIONS'

            # Prepare options list with label and value
            options_list = [
                {"label": option.value, "value": option.id} for option in category.options.filter(status=True)
            ]

            data[key] = options_list

        return Response(data)





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

    # def get(self, request, *args, **kwargs):
    #     """
    #     Fetch the entire PiBase record based on the UUID.
    #     Ignores currentStep and returns all data using a full serializer.
    #     """
    #     instance = self.get_object()
    #     # import inside if needed
    #     serializer = PiBaseRecordGetSerializer(instance)
    #     return Response(serializer.data, status=status.HTTP_200_OK)

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
            "caseStyle": data.get("case_style_data", {}).get("CaseStyle"),
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
        }

        return Response(transformed, status=status.HTTP_200_OK)
