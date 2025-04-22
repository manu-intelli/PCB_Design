from import_export import resources, fields
from masters.models import MstCategory, MstComponent
from .utility import before_save_instance_update_create_date, CustomForeignKeyWidget
from import_export.widgets import BooleanWidget


class MstCategoryResource(resources.ModelResource):    
    component_Id = fields.Field(
        column_name='component_Id',
        attribute='component_Id',
        widget=CustomForeignKeyWidget(MstComponent, field='component_name')
    )

    is_verifier = fields.Field(
        column_name='is_verifier',
        attribute='is_verifier',
        widget=BooleanWidget()
    )

    class Meta:
        model = MstCategory
        fields = ('id', 'category_name', 'component_Id', 'is_verifier')
        import_id_fields = ('category_name', 'component_Id') 
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "Categories"
        update_on_import = True
    
    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):        
        if instance:
            if not instance.created_at:                
                instance = before_save_instance_update_create_date(instance, MstCategory)

