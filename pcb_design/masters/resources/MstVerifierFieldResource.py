from import_export import resources, fields
from masters.models import MstVerifierField, MstComponent, MstCategory, MstSubCategory
from .utility import before_save_instance_update_create_date, CustomForeignKeyWidget, CustomManyToManyWidget


class MstVerifierFieldResource(resources.ModelResource):    
    component = fields.Field( column_name='component', attribute='component',
                             widget=CustomForeignKeyWidget(MstComponent, field='component_name'))
    
    category = fields.Field( column_name='category', attribute='category', 
                            widget=CustomForeignKeyWidget(MstCategory, field='category_name'))
        
    sub_category = fields.Field(column_name='sub_category', attribute='sub_category', 
                                widget=CustomManyToManyWidget(MstSubCategory, separator=',', field='id'))
    

    class Meta:
        model = MstVerifierField
        fields = ('id', 'component', 'category', 'sub_category', 'field_name')
        import_id_fields = ('id',)
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "VerifierFields"
        update_on_import = True

    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):
        """Override before_save_instance to update the created_at field."""
        if instance:
            if not instance.created_at:
                instance = before_save_instance_update_create_date(instance, MstVerifierField)
