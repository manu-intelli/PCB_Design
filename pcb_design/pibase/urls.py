from django.urls import path
from .views import (
    PiBaseComponentListView,
    PiBaseFieldCategoryListView,
    PiBaseRecordListView,
    PiBaseFieldOptionListView,
    PiBaseRecordStepOneCreateView,
    GroupedFieldOptionsView,
    PiBaseRecordStepTwoUpdateView,
)

urlpatterns = [
    path('components/', PiBaseComponentListView.as_view(), name='components-list'),
    path('field-categories/', PiBaseFieldCategoryListView.as_view(), name='fieldcategory-list'),
    path('field-options/', PiBaseFieldOptionListView.as_view(), name='field-option-list'),
    path('field-options/grouped/', GroupedFieldOptionsView.as_view(), name='grouped-field-options'),
    path('records/', PiBaseRecordListView.as_view(), name='records-list'),
    path('pi-base-record/step-one/', PiBaseRecordStepOneCreateView.as_view(), name='pi-base-record-step-one'),
    path('pi-base-records/<uuid:record_id>/', PiBaseRecordStepTwoUpdateView.as_view(), name='update-record'),
]
