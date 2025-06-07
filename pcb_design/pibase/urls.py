from django.urls import path,include
from rest_framework.routers import DefaultRouter

from .views import (
    PiBaseComponentListView,
    PiBaseFieldCategoryListView,
    PiBaseRecordListView,
    PiBaseFieldOptionListView,
    GroupedFieldOptionsView,
    CheckPiBaseRecordUniqueView,
    PiBaseRecordDetailAPIView,
    PiBaseRecordDetailAPIViewUpdate,PiBaseImageViewSet
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
    path('records/create/', PiBaseRecordDetailAPIView.as_view(), name='pibase-create'),
    path('records/update/<uuid:record_id>/', PiBaseRecordDetailAPIViewUpdate.as_view(), name='pibase-update'),
    path('', include(router.urls)),
]
