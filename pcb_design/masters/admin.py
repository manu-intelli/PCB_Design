from django.contrib import admin
from .models import MstComponent, MstCategory, MstSubCategory, MstSectionRules, \
    MstSectionGroupings, MstSubCategoryTwo, MstDesignOptions, MstConditions, MstVerifierField, \
    MstVerifierRules
from .resources import MstCategoryResource, MstComponentResource, MstSubCategoryResource,\
    MstSubCategoryTwoResource, MstDesignOptionsResource, MstSectionRulesResource, MstSectionGroupingsResource,\
    MstConditionsResource, MstVerifierFieldResource, MstVerifierRulesResource
from import_export.admin import ImportExportModelAdmin, ExportActionModelAdmin


class MstComponentAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstComponentResource]
    list_display = ('id', 'component_name', 'description', 'created_by','updated_by')  
    search_fields = ('component_name', 'description')  
    list_filter = ('component_name',)  


class MstCategoryAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstCategoryResource]
    list_display = ('id', 'category_name', 'component_Id', 'created_by','updated_by', 'is_verifier')  
    search_fields = ('category_name', 'component_Id__component_name')  
    list_filter = ('category_name', 'component_Id', 'created_by', 'is_verifier')  


class MstSubCategoryAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstSubCategoryResource]
    list_display = ('id', 'sub_category_name', 'category_Id', 'created_by','updated_by', 'is_verifier')  
    search_fields = ('sub_category_name', 'category_Id__category_name')  
    list_filter = ('sub_category_name', 'created_by', 'is_verifier')  


class MstSectionRulesAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstSectionRulesResource]
    list_display = ('id', 'rule_number', 'parameter', 'min_value', 'max_value', 'nominal', 'created_by','updated_by',)  
    search_fields = ('rule_number', 'parameter')  
    list_filter = ('rule_number', 'created_by',)  


class MstSectionGroupingsAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstSectionGroupingsResource]
    list_display = ('id', 'design_doc', 'section_name', 'created_by','updated_by',)  
    search_fields = ('design_doc', 'section_name')  
    list_filter = ('section_name', 'created_by',)   


class MstSubCategoryTwoAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstSubCategoryTwoResource]
    list_display = ('id', 'sub_2_category_name', 'sub_category_id', 'created_by','updated_by',)  
    search_fields = ('sub_2_category_name', 'sub_category_id__sub_category_name')  
    list_filter = ('sub_2_category_name', 'created_by')  


class MstDesignOptionsAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstDesignOptionsResource]
    list_display = ('id', 'desing_option_name', 'sub_category_id', 'created_by','updated_by')  
    search_fields = ('desing_option_name', 'sub_category_id__sub_category_name')  
    list_filter = ('desing_option_name', 'created_by')  


class MstConditionsAdmin(ImportExportModelAdmin, admin.ModelAdmin):    
    resource_classes = [MstConditionsResource]
    list_display = (
        'id',
        'subcategory',
        'condition_variable',
        'condition_operator',
        'condition_min_value',
        'condition_max_value',
        'comparison_variable',
        'comparison_min_value',
        'comparison_max_value',
        'comparison_operator',
    )
    
    search_fields = (
        'subcategory__sub_category_name',
        'condition_variable',
        'comparison_variable',
    )

    list_filter = ('subcategory', 'condition_variable', 'comparison_operator')


class MstVerifierFieldAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstVerifierFieldResource]
    list_display = ('id', 'component', 'category', 'field_name', 'sub_category_summary')
    search_fields = ( 'component__component_name', 'category__category_name', 'field_name',)
    list_filter = ('component', 'category')

    def sub_category_summary(self, obj):
        return ", ".join([str(sub) for sub in obj.sub_category.all()])

class MstVerifierRulesAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_classes = [MstVerifierRulesResource]
    list_display = ('id', 'verifier_field', 'design_doc', 'rule_number', 'conditional_var', 'value')
    search_fields = ('design_doc', 'rule_number', 'conditional_var', 'value')
    list_filter = ('design_doc', 'rule_number')



admin.site.register(MstComponent, MstComponentAdmin)
admin.site.register(MstCategory, MstCategoryAdmin)
admin.site.register(MstSubCategory, MstSubCategoryAdmin)
admin.site.register(MstSectionRules, MstSectionRulesAdmin)
admin.site.register(MstSectionGroupings, MstSectionGroupingsAdmin)
admin.site.register(MstSubCategoryTwo, MstSubCategoryTwoAdmin)
admin.site.register(MstDesignOptions, MstDesignOptionsAdmin)
admin.site.register(MstConditions, MstConditionsAdmin)
admin.site.register(MstVerifierField, MstVerifierFieldAdmin)
admin.site.register(MstVerifierRules, MstVerifierRulesAdmin)
