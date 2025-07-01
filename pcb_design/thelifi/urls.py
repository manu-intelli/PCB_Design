from django.urls import path
from .views import (
    FilterUploadView,
    KPI_CalculationAPIView,
    SubmissionFilesListAPIView,
    SParameterPlotAPIView,
    StatisticalAndHistogramPlotAPIView,
    DownloadPlotsZipAPIView,
    SubmissionRecordListAPIView,
    SubmissionRecordDetailAPIView,
)

urlpatterns = [
    path("api/upload/", FilterUploadView.as_view(), name="filter-upload"),
    path("api/calculate-kpi/", KPI_CalculationAPIView.as_view(), name="kpi-calculation"),
    path(
        "api/generate-s-parameter-plots/",
        SParameterPlotAPIView.as_view(),
        name="generate-s-parameter-plots",
    ),
    path(
        "api/generate-statistical-histogram-plots/",
        StatisticalAndHistogramPlotAPIView.as_view(),
        name="generate-statistical-histogram-plots",
    ),
    path(
        "api/submission-files-list/",
        SubmissionFilesListAPIView.as_view(),
        name="submission-files-list",
    ),
    path(
        "api/download-plots/",
        DownloadPlotsZipAPIView.as_view(),
        name="download-plots-zip",
    ),
    path("records/", SubmissionRecordListAPIView.as_view(), name="record-list"),
    path(
        "records/<int:pk>/",
        SubmissionRecordDetailAPIView.as_view(),
        name="record-detail",
    ),
]
