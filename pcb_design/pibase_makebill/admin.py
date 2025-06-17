from django.contrib import admin
from .models import MakeBillRecord, MakeBillStatus

@admin.register(MakeBillRecord)
class MakeBillRecordAdmin(admin.ModelAdmin):
    """
    Admin configuration for MakeBillRecord model.
    """
    list_display = ('model_name', 'op_number', 'opu_number', 'edu_number', 'status', 'created_by', 'created_at')
    list_filter = ('status', 'created_by', 'created_at')
    search_fields = ('model_name', 'op_number', 'opu_number', 'edu_number')

@admin.register(MakeBillStatus)
class MakeBillStatusAdmin(admin.ModelAdmin):
    """
    Admin configuration for MakeBillStatus model.
    """
    list_display = ('status_code', 'description', 'created_at')
    list_filter = ('status_code',)
    search_fields = ('description',)
