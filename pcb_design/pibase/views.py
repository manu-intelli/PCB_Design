from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from authentication.custom_permissions import IsAuthorized
from authentication.custom_authentication import CustomJWTAuthentication
from django.http import Http404
from .models import PiBaseComponent, PiBaseFieldCategory, PiBaseRecord, PiBaseFieldOption
from .serializers import (
    PiBaseComponentSerializer,
    PiBaseFieldCategorySerializer,
    PiBaseRecordSerializer,
    PiBaseFieldOptionSerializer,
    PiBaseRecordStepOneSerializer,
    PiBaseRecordStepTwoSerializer,
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


class PiBaseRecordListView(generics.ListAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordSerializer


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


class PiBaseRecordStepTwoUpdateView(generics.UpdateAPIView):
    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseRecordStepTwoSerializer
    lookup_field = 'id'  # The actual primary key

    def get_object(self):
        record_uuid = self.kwargs['record_id']
        for obj in self.get_queryset():
            generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f'PiBase-{obj.id}')
            if str(generated_uuid) == str(record_uuid):
                return obj
        raise Http404("Record not found")