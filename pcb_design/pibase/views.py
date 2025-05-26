from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import PiBaseComponent, PiBaseFieldCategory, PiBaseRecord, PiBaseFieldOption
from .serializers import (
    PiBaseComponentSerializer,
    PiBaseFieldCategorySerializer,
    PiBaseRecordSerializer,
    PiBaseFieldOptionSerializer,
    PiBaseRecordStepOneSerializer
)

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
    # permission_classes = [permissions.IsAuthenticated]  # Optional, use if auth is required

    def perform_create(self, serializer):
        # Set created_by if needed (optional)
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
