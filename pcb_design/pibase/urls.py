from django.urls import path
from .views import (
    PiBaseComponentListView,
    PiBaseFieldCategoryListView,
    PiBaseRecordListView,
    PiBaseFieldOptionListView,
    PiBaseRecordStepOneCreateView,
    GroupedFieldOptionsView,
    PiBaseRecordPartialUpdateView,
    CheckPiBaseRecordUniqueView,
    PiBaseRecordDetailAPIView,
    PiBaseRecordDetailAPIViewUpdate
)

urlpatterns = [
    path('components/', PiBaseComponentListView.as_view(), name='components-list'),
    path('field-categories/', PiBaseFieldCategoryListView.as_view(), name='fieldcategory-list'),
    path('field-options/', PiBaseFieldOptionListView.as_view(), name='field-option-list'),
    path('field-options/grouped/', GroupedFieldOptionsView.as_view(), name='grouped-field-options'),
    path('records/', PiBaseRecordListView.as_view(), name='records-list'),
    path('pi-base-record/step-one/', PiBaseRecordStepOneCreateView.as_view(), name='pi-base-record-step-one'),
    path('pi-base-records/<uuid:record_id>/', PiBaseRecordPartialUpdateView.as_view(), name='update-record'),
    path('pibase/check-unique/', CheckPiBaseRecordUniqueView.as_view(), name='check-pibase-unique'),
    path('pibase/create/', PiBaseRecordDetailAPIView.as_view(), name='pibase-create'),
    path('pibase/update/<int:id>', PiBaseRecordDetailAPIViewUpdate.as_view(), name='pibase-update'),
]
