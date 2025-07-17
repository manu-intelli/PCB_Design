"""
Views for the pibase app.

These views provide API endpoints for listing, retrieving, creating, and updating PiBase models,
as well as for grouped dropdown options and uniqueness checks. All endpoints use appropriate
authentication and permissions, and include logging for success, warning, and error events.
"""

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
    """
    API endpoint to list all PiBaseComponent records.
    """
    queryset = PiBaseComponent.objects.all()
    serializer_class = PiBaseComponentSerializer
    pagination_class = None

    def list(self, request, *args, **kwargs):
        """
        Handles GET requests to list all PiBaseComponent records.
        Logs success and error events.
        """
        try:
            response = super().list(request, *args, **kwargs)
            pi_base_logs.info("Successfully listed all PiBaseComponent records.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error listing PiBaseComponent records: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# =====================================================================================================================================

class PiBaseFieldCategoryListView(generics.ListAPIView):
    """
    API endpoint to list all PiBaseFieldCategory records.
    """
    queryset = PiBaseFieldCategory.objects.all()
    serializer_class = PiBaseFieldCategorySerializer
    pagination_class = None

    def list(self, request, *args, **kwargs):
        """
        Handles GET requests to list all PiBaseFieldCategory records.
        Logs success and error events.
        """
        try:
            response = super().list(request, *args, **kwargs)
            pi_base_logs.info("Successfully listed all PiBaseFieldCategory records.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error listing PiBaseFieldCategory records: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# =====================================================================================================================================

class PiBaseFieldOptionListView(generics.ListAPIView):
    """
    API endpoint to list all PiBaseFieldOption records.
    """
    queryset = PiBaseFieldOption.objects.all()
    serializer_class = PiBaseFieldOptionSerializer
    pagination_class = None

    def list(self, request, *args, **kwargs):
        """
        Handles GET requests to list all PiBaseFieldOption records.
        Logs success and error events.
        """
        try:
            response = super().list(request, *args, **kwargs)
            pi_base_logs.info("Successfully listed all PiBaseFieldOption records.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error listing PiBaseFieldOption records: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# =====================================================================================================================================

from rest_framework.pagination import PageNumberPagination

class PiBaseRecordPagination(PageNumberPagination):
    """
    Pagination class for PiBaseRecord list endpoints.
    Defines the page size and allows clients to specify it via a query parameter.
    """
    page_size = 10
    page_size_query_param = (
        "page_size"  # optional, allow client to set page size in query param
    )
    max_page_size = 100

# =====================================================================================================================================

from django_filters.rest_framework import DjangoFilterBackend

class PiBaseRecordListView(generics.ListAPIView):
    """
    API endpoint to list PiBaseRecord records for the current user.
    Supports filtering, searching, and ordering.
    """
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
        """
        Returns PiBaseRecord objects created by the current user.
        Handles exceptions and logs errors.
        """
        try:
            queryset = PiBaseRecord.objects.filter(created_by=self.request.user)
            return queryset
        except Exception as e:
            pi_base_logs.error(f"Error filtering PiBaseRecord for user {self.request.user}: {e}")
            return PiBaseRecord.objects.none()

    def list(self, request, *args, **kwargs):
        """
        Handles GET requests to list PiBaseRecord records for the current user.
        Logs success and error events.
        """
        try:
            response = super().list(request, *args, **kwargs)
            pi_base_logs.info(f"Successfully listed PiBaseRecord records for user {request.user}.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error listing PiBaseRecord records: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# =====================================================================================================================================

class PiBaseImageViewSet(viewsets.ModelViewSet):
    """
    API endpoint for CRUD operations on PiBaseImage records.
    """
    queryset = PiBaseImage.objects.all().order_by("-created_at")
    serializer_class = PiBaseImageSerializer

    def list(self, request, *args, **kwargs):
        """
        Handles GET requests to list all PiBaseImage records.
        Logs success and error events.
        """
        try:
            response = super().list(request, *args, **kwargs)
            pi_base_logs.info("Successfully listed all PiBaseImage records.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error listing PiBaseImage records: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def create(self, request, *args, **kwargs):
        """
        Handles POST requests to create a new PiBaseImage record.
        Logs success and error events.
        """
        try:
            response = super().create(request, *args, **kwargs)
            pi_base_logs.info("Successfully created a new PiBaseImage record.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error creating PiBaseImage record: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def update(self, request, *args, **kwargs):
        """
        Handles PUT/PATCH requests to update an existing PiBaseImage record.
        Logs success and error events.
        """
        try:
            response = super().update(request, *args, **kwargs)
            pi_base_logs.info("Successfully updated a PiBaseImage record.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error updating PiBaseImage record: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def destroy(self, request, *args, **kwargs):
        """
        Handles DELETE requests to delete a PiBaseImage record.
        Logs success and error events.
        """
        try:
            response = super().destroy(request, *args, **kwargs)
            pi_base_logs.info("Successfully deleted a PiBaseImage record.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error deleting PiBaseImage record: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
        Logs information and errors during the process.
        """
        data = {}
        try:
            self.add_pibase_field_options(data)
            self.add_mst_category_options(data)
            pi_base_logs.info("Successfully fetched grouped field options.")
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            pi_base_logs.error(f"Error fetching grouped field options: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def add_pibase_field_options(self, data):
        """
        Adds options from PiBaseFieldCategory and its related options to the response dictionary.
        Categories and their active options are formatted into {LABEL, VALUE} pairs.
        Handles exceptions during category processing.
        """
        try:
            categories = PiBaseFieldCategory.objects.filter(status=True).prefetch_related("options")
            for cat in categories:
                key = f"{cat.name.upper().replace(' ', '_')}_OPTIONS"
                try:
                    data[key] = [
                        {"label": opt.value, "value": opt.id}
                        for opt in cat.options.all() if opt.status
                    ]
                except Exception as inner_err:
                    pi_base_logs.warning(f"Error processing options for category '{cat.name}': {str(inner_err)}")
                    data[key] = []
        except Exception as e:
            pi_base_logs.error(f"[PiBase] Error fetching PiBaseFieldCategory data: {str(e)}")
            # Optionally, you might raise this exception further or return an error response

    def add_mst_category_options(self, data):
        """
        Adds mapped MstCategory and related MstSubCategory options to the response dictionary.
        Each set is appended with an 'Others' item with a static fallback value.
        Handles exceptions during category lookup and sub-category fetching.
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
                pi_base_logs.error(f"[MstSubCategory] Error fetching options for '{name}': {str(e)}")
                data[key] = []

# =====================================================================================================================================

class CheckPiBaseRecordUniqueView(APIView):
    """
    API endpoint to check uniqueness of PiBaseRecord fields in the database.
    Accepts op_number, opu_number, edu_number, and model_name in POST data.
    Returns uniqueness status and message for each field.
    """

    @swagger_auto_schema(request_body=PiBaseRecordUniquenessSerializer)
    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to check the uniqueness of specified PiBaseRecord fields.
        Validates input data and checks database for existing records.
        Logs success, warning, and error events.
        """
        try:
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

                pi_base_logs.info("Successfully checked uniqueness for PiBaseRecord fields.")
                return Response(results, status=status.HTTP_200_OK)

            pi_base_logs.warning(f"Invalid data received for uniqueness check: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            pi_base_logs.error(f"Error in CheckPiBaseRecordUniqueView.post: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# =====================================================================================================================================

class PiBaseRecordCreateAPIView(generics.CreateAPIView):
    """
    API endpoint to create a new PiBaseRecord.
    """
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordFullSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def create(self, request, *args, **kwargs):
        """
        Handles POST requests to create a new PiBaseRecord.
        Logs success and error events.
        """
        try:
            response = super().create(request, *args, **kwargs)
            pi_base_logs.info(f"Successfully created a new PiBaseRecord by user {request.user}.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error creating PiBaseRecord: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PiBaseRecordUpdateAPIView(generics.UpdateAPIView):
    """
    API endpoint to update an existing PiBaseRecord by UUID.
    """
    serializer_class = PiBaseRecordFullSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get_queryset(self):
        """
        Returns the queryset of all PiBaseRecord objects.
        """
        return PiBaseRecord.objects.all()

    def get_object(self):
        """
        Retrieves a single PiBaseRecord object based on the UUID provided in the URL.
        It iterates through all records and generates a UUID for each to find a match.
        Raises Http404 if no record is found or an unexpected error occurs during retrieval.
        """
        try:
            record_uuid = self.kwargs.get("record_id")
            for obj in self.get_queryset():
                generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}")
                if str(generated_uuid) == str(record_uuid):
                    return obj
            pi_base_logs.error(f"Record not found for update: {record_uuid}")
            raise Http404("Record not found")
        except Http404:
            # Re-raise Http404 as it's an expected outcome for "not found"
            raise
        except Exception as e:
            pi_base_logs.error(f"Error in PiBaseRecordUpdateAPIView.get_object: {e}")
            # Raise a more generic Http404 for unexpected errors during retrieval
            raise Http404("Record not found due to an internal error.")

    def update(self, request, *args, **kwargs):
        """
        Handles PUT/PATCH requests to update an existing PiBaseRecord.
        Logs success and error events.
        """
        try:
            response = super().update(request, *args, **kwargs)
            pi_base_logs.info(f"Successfully updated PiBaseRecord {kwargs.get('record_id')}.")
            return response
        except Exception as e:
            pi_base_logs.error(f"Error updating PiBaseRecord {kwargs.get('record_id')}: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PiBaseRecordRetrieveAPIView(generics.RetrieveAPIView):
    """
    API endpoint to retrieve a PiBaseRecord by UUID.
    """
    serializer_class = PiBaseRecordGetSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get_queryset(self):
        """
        Returns the queryset of all PiBaseRecord objects.
        """
        return PiBaseRecord.objects.all()

    def get_object(self):
        """
        Retrieves a single PiBaseRecord object based on the UUID provided in the URL.
        It iterates through all records and generates a UUID for each to find a match.
        Raises Http404 if no record is found or an unexpected error occurs during retrieval.
        """
        try:
            record_uuid = self.kwargs.get("record_id")
            for obj in self.get_queryset():
                generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}")
                if str(generated_uuid) == str(record_uuid):
                    pi_base_logs.info(f"Successfully retrieved PiBaseRecord {record_uuid}.")
                    return obj
            pi_base_logs.error(f"Record not found for retrieve: {record_uuid}")
            raise Http404("Record not found")
        except Http404:
            # Re-raise Http404 as it's an expected outcome for "not found"
            raise
        except Exception as e:
            pi_base_logs.error(f"Error in PiBaseRecordRetrieveAPIView.get_object: {e}")
            # Raise a more generic Http404 for unexpected errors during retrieval
            raise Http404("Record not found due to an internal error.")

# =====================================================================================================================================