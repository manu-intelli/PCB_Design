from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import (
    PiBaseFieldCategory,
    PiBaseFieldOption,
    PiBaseComponent,
    PiBaseStatus,
    PiBaseRecord,
    PiBaseImage,  # ✅ Add import
)

# Admin class for PiBaseFieldCategory model
@admin.register(PiBaseFieldCategory)
class PiBaseFieldCategoryAdmin(ImportExportModelAdmin):
    # Displays field category details in admin list view
    list_display = ("name", "input_type", "status", "created_at", "updated_at")
    # Enables searching by field name
    search_fields = ("name",)
    # Adds filtering by status and input type
    list_filter = ("status", "input_type")
    # Orders by most recent creation date
    ordering = ("-created_at",)

# Admin class for PiBaseFieldOption model
@admin.register(PiBaseFieldOption)
class PiBaseFieldOptionAdmin(ImportExportModelAdmin):
    # Displays field option details
    list_display = ("category", "value", "status", "created_at", "updated_at")
    # Enables search on value and related category name
    search_fields = ("value", "category__name")
    # Adds filter options for status and category
    list_filter = ("status", "category")

# Admin class for PiBaseComponent model
@admin.register(PiBaseComponent)
class PiBaseComponentAdmin(ImportExportModelAdmin):
    # Displays component information
    list_display = ("name", "status", "created_at", "updated_at")
    # Search by component name
    search_fields = ("name",)
    # Filter by component status
    list_filter = ("status",)

# Admin class for PiBaseStatus model
@admin.register(PiBaseStatus)
class PiBaseStatusAdmin(ImportExportModelAdmin):
    # Displays status codes and descriptions
    list_display = ("status_code", "description", "created_at", "updated_at")
    # Search based on description
    search_fields = ("description",)
    # Order by status code
    ordering = ("status_code",)

# Admin class for PiBaseImage model
@admin.register(PiBaseImage)
class PiBaseImageAdmin(ImportExportModelAdmin):
    # Displays image metadata in admin
    list_display = ("image_type", "image_file", "cookies", "created_at", "updated_at")
    # Enables search by image ID
    search_fields = ("id",)
    # Makes timestamps read-only
    readonly_fields = ("created_at", "updated_at")

# Admin class for PiBaseRecord model
@admin.register(PiBaseRecord)
class PiBaseRecordAdmin(ImportExportModelAdmin):
    # Displays main record information
    list_display = (
        "model_name",
        "op_number",
        "status",
        "created_by",
        "updated_by",
        "created_at",
        "updated_at",
    )
    # Searchable fields in admin
    search_fields = ("model_name", "op_number", "edu_number", "opu_number")
    # Filter records based on status
    list_filter = ("status",)  # ✅ Correct tuple syntax
    # Enables autocomplete for related foreign/many-to-many fields
    autocomplete_fields = (
        "model_family",
        "bottom_solder_mask",
        "half_moon_requirement",
        "via_holes_requirement",
        "signal_launch_type",
        "cover_type",
        "design_rule_violation",
        "status",
        "created_by",
        "updated_by",
        "components",
    )
    # Makes timestamps read-only
    readonly_fields = ("created_at", "updated_at")
    # Orders by most recent creation
    ordering = ("-created_at",)
