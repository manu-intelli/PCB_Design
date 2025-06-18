"""
Views for the pibase_makebill app.

Provides API endpoints for listing PiBaseRecord records for the current user,
with filtering, searching, ordering, and error handling.
"""

from rest_framework import generics, status, filters
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from django.http import Http404
import uuid

from pibase.views import PiBaseRecordPagination
from pibase.models import PiBaseRecord
from .serializers import PiBaseToMakeBillRecordSerializer,MakeBillRecordGetSerializer
from authentication.custom_permissions import IsAuthorized
from authentication.custom_authentication import CustomJWTAuthentication
from . import make_bill_logs

class PiBaseToMakeBilRecordListView(generics.ListAPIView):
    """
    API endpoint to list PiBaseRecord records for the current user with status=2.
    Supports filtering, searching, and ordering.
    """
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseToMakeBillRecordSerializer
    pagination_class = PiBaseRecordPagination  # Set your pagination class if needed
    permission_classes = [IsAuthenticated]
    
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_fields = ["status"]
    search_fields = ["model_name", "op_number", "opu_number", "edu_number"]
    ordering_fields = [
        "id",
        "model_name",
        "op_number",
        "opu_number",
        "edu_number",
        "created_at",
        "revision_number",
    ]
    ordering = ["-created_at"]

    def get_queryset(self):
        """
        Returns PiBaseRecord objects created by the current user and with status=2.
        Handles exceptions and logs errors.
        """
        try:
            return PiBaseRecord.objects.filter(status=2)
        except Exception as e:
            make_bill_logs.error(f"Error filtering PiBaseRecord for user {self.request.user}: {e}")
            return PiBaseRecord.objects.none()

    def list(self, request, *args, **kwargs):
        """
        Returns a list of PiBaseRecord records for the current user.
        Logs success and error events.
        """
        try:
            response = super().list(request, *args, **kwargs)
            make_bill_logs.info(f"Successfully listed PiBaseRecord records for user {request.user}.")
            return response
        except Exception as e:
            make_bill_logs.error(f"Error listing PiBaseRecord records: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class MakeBillRetrieveAPIView(generics.RetrieveAPIView):
    """
    API endpoint to retrieve a PiBaseRecord by UUID.
    """
    serializer_class = MakeBillRecordGetSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get_queryset(self):
        return PiBaseRecord.objects.all()

    def get_object(self):
        try:
            record_uuid = self.kwargs.get("record_id")
            for obj in self.get_queryset():
                generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}")
                if str(generated_uuid) == str(record_uuid):
                    make_bill_logs.info(f"Successfully retrieved PiBaseRecord {record_uuid}.")
                    return obj
            make_bill_logs.error("Record not found for retrieve.")
            raise Http404("Record not found")
        except Exception as e:
            make_bill_logs.error(f"Error in PiBaseRecordRetrieveAPIView.get_object: {e}")
            raise Http404("Record not found")