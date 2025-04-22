from import_export import resources, fields
from masters.models import MstConditions, MstSubCategory
from .utility import before_save_instance_update_create_date, CustomForeignKeyWidget


class MstConditionsResource(resources.ModelResource):
    subcategory = fields.Field(
        column_name='subcategory',
        attribute='subcategory',
        widget=CustomForeignKeyWidget(MstSubCategory, field='id')
    )
    
    class Meta:
        model = MstConditions
        fields = (
            'id', 
            'subcategory', 
            'condition_variable', 
            'condition_operator', 
            'condition_min_value', 
            'condition_max_value', 
            'comparison_variable',
            'comparison_min_value', 
            'comparison_max_value', 
            'comparison_operator'
        )
        import_id_fields = ('id',)  # ID will be used for updating or creating entries
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "Conditions"
        update_on_import = True
    
    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):
        """Override before_save_instance to update the created_at field."""
        if instance:
            if not instance.created_at:
                instance = before_save_instance_update_create_date(instance, MstConditions)
