from django.contrib import admin
from .models import FilterSubmission

@admin.register(FilterSubmission)
class FilterSubmissionAdmin(admin.ModelAdmin):
    list_display = ('id', 'model_number', 'edu_number', 'filter_type', 'folder_path', 'created_at')
    readonly_fields = ('created_at', 'list_files_in_folder')
    search_fields = ('model_number', 'edu_number', 'filter_type')
    list_filter = ('filter_type',)
