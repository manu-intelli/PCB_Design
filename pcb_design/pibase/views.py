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
    PiBaseRecordGetSerializer
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

    def get_queryset(self):
        # Return only records created by the current user
        return PiBaseRecord.objects.filter(created_by=self.request.user)


# =====================================================================================================================================


class PiBaseImageViewSet(viewsets.ModelViewSet):
    queryset = PiBaseImage.objects.all().order_by("-created_at")
    serializer_class = PiBaseImageSerializer


# =====================================================================================================================================

class GroupedFieldOptionsView(APIView):
    """
    API endpoint that returns grouped dropdown options for:
    - PiBaseFieldCategory options (e.g., material types, layer types)
    - MstCategory + MstSubCategory based groups (e.g., Copper Thickness)

    Response Format:
    {
        "CATEGORY_NAME_OPTIONS": [{"label": "Option Label", "value": 1}, ...],
        ...
    }
    """

    CATEGORY_MAP = {
        "Dielectric material": ("DIELECTRIC_MATERIAL_OPTIONS", 10000),
        "Copper Thickness": ("COPPER_THICKNESS_OPTIONS", 10001),
        "Dielectric Thickness": ("DIELECTRIC_THICKNESS_OPTIONS", 10002),
    }

    def get(self, request):
        """
        GET method to fetch grouped field options from PiBaseFieldCategory and MstCategory data sources.
        """
        data = {}
        self.add_pibase_field_options(data)
        self.add_mst_category_options(data)
        return Response(data, status=status.HTTP_200_OK)

    def add_pibase_field_options(self, data):
        """
        Adds options from PiBaseFieldCategory and its related options to the response dictionary.
        Categories and their active options are formatted into {LABEL, VALUE} pairs.
        """
        try:
            categories = PiBaseFieldCategory.objects.filter(status=True).prefetch_related("options")
            for cat in categories:
                try:
                    key = f"{cat.name.upper().replace(' ', '_')}_OPTIONS"
                    data[key] = [
                        {"label": opt.value, "value": opt.id}
                        for opt in cat.options.all() if opt.status
                    ]
                except Exception as inner_err:
                    pi_base_logs.warning(f"Error processing category '{cat.name}': {str(inner_err)}")
                    data[key] = []
        except Exception as e:
            pi_base_logs.error(f"[PiBase] Error fetching categories: {str(e)}")

    def add_mst_category_options(self, data):
        """
        Adds mapped MstCategory and related MstSubCategory options to the response dictionary.
        Each set is appended with an 'Others' item with a static fallback value.
        """
        for name, (key, other_value) in self.CATEGORY_MAP.items():
            try:
                category = MstCategory.objects.filter(category_name__iexact=name).only("id").first()
                if not category:
                    pi_base_logs.warning(f"[MstCategory] '{name}' not found.")
                    data[key] = []
                    continue

                options = MstSubCategory.objects.filter(category_Id=category.id).only("id", "sub_category_name")
                data[key] = [{"label": sub.sub_category_name, "value": sub.id} for sub in options]
                data[key].append({"label": "Others", "value": other_value})

            except Exception as e:
                pi_base_logs.error(f"[MstSubCategory] Error for '{name}': {str(e)}")
                data[key] = []


                
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
    serializer_class = PiBaseRecordGetSerializer
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

