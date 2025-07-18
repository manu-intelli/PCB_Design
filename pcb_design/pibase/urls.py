"""
URL configuration for the pibase app.

Defines API endpoints for listing, retrieving, creating, and updating PiBase models,
as well as endpoints for grouped dropdown options, uniqueness checks, and image uploads.

Endpoints:
    - components/                : List all PiBaseComponent records.
    - field-categories/          : List all PiBaseFieldCategory records.
    - field-options/             : List all PiBaseFieldOption records.
    - field-options/grouped/     : Get grouped dropdown options for fields.
    - records/                   : List PiBaseRecord records for the current user.
    - check-unique/              : Check uniqueness of PiBaseRecord fields.
    - records/create/            : Create a new PiBaseRecord.
    - records/update/<uuid>/     : Update an existing PiBaseRecord by UUID.
    - records/get/<uuid>/        : Retrieve a PiBaseRecord by UUID.
    - uploadimages/              : CRUD operations for PiBaseImage records (via router).
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    PiBaseComponentListView,
    PiBaseFieldCategoryListView,
    PiBaseRecordListView,
    PiBaseFieldOptionListView,
    GroupedFieldOptionsView,
    CheckPiBaseRecordUniqueView,
    PiBaseImageViewSet,
    PiBaseRecordCreateAPIView,
    PiBaseRecordUpdateAPIView,
    PiBaseRecordRetrieveAPIView,
)

router = DefaultRouter()
router.register(r'uploadimages', PiBaseImageViewSet, basename='pibase-image')

urlpatterns = [
    path('components/', PiBaseComponentListView.as_view(), name='components-list'),
    path('field-categories/', PiBaseFieldCategoryListView.as_view(), name='fieldcategory-list'),
    path('field-options/', PiBaseFieldOptionListView.as_view(), name='field-option-list'),
    path('field-options/grouped/', GroupedFieldOptionsView.as_view(), name='grouped-field-options'),
    path('records/', PiBaseRecordListView.as_view(), name='records-list'),
    path('check-unique/', CheckPiBaseRecordUniqueView.as_view(), name='check-pibase-unique'),
    path('records/create/', PiBaseRecordCreateAPIView.as_view(), name='pibase-create'),
    path('records/update/<uuid:record_id>/', PiBaseRecordUpdateAPIView.as_view(), name='pibase-update'),
    path('records/get/<uuid:record_id>/', PiBaseRecordRetrieveAPIView.as_view(), name='pibase-get'),
    path('', include(router.urls)),
]
