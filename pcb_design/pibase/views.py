from rest_framework import generics, permissions, status, filters, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from authentication.custom_permissions import IsAuthorized
from authentication.custom_authentication import CustomJWTAuthentication
from django.http import Http404
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .models import (
    PiBaseComponent,
    PiBaseFieldCategory,
    PiBaseRecord,
    PiBaseFieldOption,
    PiBaseImage,
)
from .serializers import (
    PiBaseComponentSerializer,
    PiBaseFieldCategorySerializer,
    PiBaseRecordSerializer,
    PiBaseFieldOptionSerializer,
    PiBaseRecordUniquenessSerializer,
    PiBaseRecordFullSerializer,
    PiBaseImageSerializer,
)
import uuid

from masters.models import MstCategory, MstSubCategory

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
    page_size_query_param = (
        "page_size"  # optional, allow client to set page size in query param
    )
    max_page_size = 100


# =====================================================================================================================================

from django_filters.rest_framework import DjangoFilterBackend

class PiBaseRecordListView(generics.ListAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordSerializer
    pagination_class = PiBaseRecordPagination

    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_fields = ["status"]  # For exact filtering (e.g., ?status=2)
    search_fields = ["model_name", "op_number", "opu_number", "edu_number"]  # For text search
    ordering_fields = [
        "id",
        "model_name",
        "op_number",
        "opu_number",
        "edu_number",
        "created_at",
        "revision_number",
    ]
    ordering = ["-created_at"]  # Default ordering


# =====================================================================================================================================


class PiBaseImageViewSet(viewsets.ModelViewSet):
    queryset = PiBaseImage.objects.all().order_by("-created_at")
    serializer_class = PiBaseImageSerializer


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
            categories = PiBaseFieldCategory.objects.filter(
                status=True
            ).prefetch_related("options")
            for category in categories:
                key = category.name.upper().replace(" ", "_") + "_OPTIONS"
                options_list = [
                    {"label": option.value, "value": option.id}
                    for option in category.options.filter(status=True)
                ]
                data[key] = options_list
        except Exception as e:
            pi_base_logs.error(f"Error fetching PiBaseFieldCategory options: {str(e)}")
            return Response(
                {"detail": "Error retrieving field category options."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Custom MstCategory-based groups to include
        category_names_to_include = {
            "Dielectric material": "DIELECTRIC_MATERIAL_OPTIONS",
            "Copper Thickness": "COPPER_THICKNESS_OPTIONS",
            "Dielectric Thickness": "DIELECTRIC_THICKNESS_OPTIONS",
        }

        for cat_name, response_key in category_names_to_include.items():
            try:
                category = MstCategory.objects.get(category_name__iexact=cat_name)
                subcategories = MstSubCategory.objects.filter(category_Id=category.id)
                data[response_key] = [
                    {"label": sub.sub_category_name, "value": sub.id}
                    for sub in subcategories
                ]
                # Add "Others" option only for DIELECTRIC_MATERIAL_OPTIONS
                if response_key == "DIELECTRIC_MATERIAL_OPTIONS":
                    data[response_key].append({"label": "Others", "value": 10000})

            except MstCategory.DoesNotExist:
                pi_base_logs.warning(f"MstCategory not found for: {cat_name}")
                data[response_key] = []
            except Exception as e:
                pi_base_logs.error(
                    f"Error fetching subcategories for {cat_name}: {str(e)}"
                )
                data[response_key] = []

        return Response(data, status=status.HTTP_200_OK)


# =====================================================================================================================================


class CheckPiBaseRecordUniqueView(APIView):

    @swagger_auto_schema(request_body=PiBaseRecordUniquenessSerializer)
    def post(self, request, *args, **kwargs):
        serializer = PiBaseRecordUniquenessSerializer(data=request.data)
        if serializer.is_valid():
            op_number = serializer.validated_data.get("op_number")
            opu_number = serializer.validated_data.get("opu_number")
            edu_number = serializer.validated_data.get("edu_number")
            model_name = serializer.validated_data.get("model_name")

            results = {
                "op_number": {
                    "unique": not PiBaseRecord.objects.filter(op_number=op_number).exists(),
                },
                "opu_number": {
                    "unique": not PiBaseRecord.objects.filter(opu_number=opu_number).exists(),
                },
                "edu_number": {
                    "unique": not PiBaseRecord.objects.filter(edu_number=edu_number).exists(),
                },
                "model_name": {
                    "unique": not PiBaseRecord.objects.filter(
                        model_name=model_name
                    ).exists(),
                },
            }

            # Add helpful message per field
            for field, result in results.items():
                result["message"] = (
                    f"{field} is unique."
                    if result["unique"]
                    else f"{field} already exists."
                )

            return Response(results, status=status.HTTP_200_OK)

        # If input data itself is invalid
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# =====================================================================================================================================


class PiBaseRecordCreateAPIView(generics.CreateAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordFullSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]


class PiBaseRecordUpdateAPIView(generics.UpdateAPIView):
    serializer_class = PiBaseRecordFullSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get_queryset(self):
        return PiBaseRecord.objects.all()

    def get_object(self):
        record_uuid = self.kwargs.get("record_id")
        for obj in self.get_queryset():
            generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}")
            if str(generated_uuid) == str(record_uuid):
                return obj
        raise Http404("Record not found")


class PiBaseRecordRetrieveAPIView(generics.RetrieveAPIView):
    serializer_class = PiBaseRecordFullSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get_queryset(self):
        return PiBaseRecord.objects.all()

    def get_object(self):
        record_uuid = self.kwargs.get("record_id")
        for obj in self.get_queryset():
            generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}")
            if str(generated_uuid) == str(record_uuid):
                return obj
        raise Http404("Record not found")

# =====================================================================================================================================

