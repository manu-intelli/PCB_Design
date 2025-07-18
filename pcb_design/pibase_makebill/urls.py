from django.urls import path
from .views import (
    PiBaseToMakeBilRecordListView,
    MakeBillRetrieveAPIView,
    MakeBillListAPIView,
    MakeBillCreateAPIView,
    MakeBillGetAPIView,
    MakeBillUpdateAPIView,
    MakeBillDeleteAPIView,
)

urlpatterns = [
    path(
        "pibaseTomakebill-records/",
        PiBaseToMakeBilRecordListView.as_view(),
        name="records-list",
    ),
    path(
        "makebill-get-record/<uuid:record_id>/",
        MakeBillRetrieveAPIView.as_view(),
        name="makebill-record-retrieve",
    ),
    path("", MakeBillListAPIView.as_view(), name="makebill-list"),
    path("create/", MakeBillCreateAPIView.as_view(), name="makebill-create"),
    path("<uuid:record_id>/", MakeBillGetAPIView.as_view(), name="makebill-detail"),  # Fetch a single makebill by id
    path("<uuid:record_id>/update/", MakeBillUpdateAPIView.as_view(), name="makebill-update"),
    path("<int:record_id>/delete/", MakeBillDeleteAPIView.as_view(), name="makebill-delete"),
]
