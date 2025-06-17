from django.urls import path
from .views import PiBaseToMakeBilRecordListView, MakeBillRetrieveAPIView

urlpatterns = [
    path('pibaseTomakebill-records/', PiBaseToMakeBilRecordListView.as_view(), name='records-list'),
    path('makebill-get-record/<uuid:record_id>/', MakeBillRetrieveAPIView.as_view(), name='makebill-record-retrieve'),
]