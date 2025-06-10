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


@admin.register(PiBaseFieldCategory)
class PiBaseFieldCategoryAdmin(ImportExportModelAdmin):
    list_display = ("name", "input_type", "status", "created_at", "updated_at")
    search_fields = ("name",)
    list_filter = ("status", "input_type")
    ordering = ("-created_at",)


@admin.register(PiBaseFieldOption)
class PiBaseFieldOptionAdmin(ImportExportModelAdmin):
    list_display = ("category", "value", "status", "created_at", "updated_at")
    search_fields = ("value", "category__name")
    list_filter = ("status", "category")


@admin.register(PiBaseComponent)
class PiBaseComponentAdmin(ImportExportModelAdmin):
    list_display = ("name", "status", "created_at", "updated_at")
    search_fields = ("name",)
    list_filter = ("status",)


@admin.register(PiBaseStatus)
class PiBaseStatusAdmin(ImportExportModelAdmin):
    list_display = ("status_code", "description", "created_at", "updated_at")
    search_fields = ("description",)
    ordering = ("status_code",)


@admin.register(PiBaseImage)
class PiBaseImageAdmin(ImportExportModelAdmin):
    list_display = ("image_type", "image_file","cookies", "created_at", "updated_at")
    search_fields = ("id",)
    readonly_fields = ("created_at", "updated_at")


@admin.register(PiBaseRecord)
class PiBaseRecordAdmin(ImportExportModelAdmin):
    list_display = (
        "model_name",
        "op_number",
        "status",
        "created_by",
        "updated_by",
        "created_at",
        "updated_at",
    )
    search_fields = ("model_name", "op_number", "edu_number", "opu_number")
    list_filter = ("status",)  # ✅ Correct tuple syntax
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
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-created_at",)

