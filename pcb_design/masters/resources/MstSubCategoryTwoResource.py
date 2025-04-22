from import_export import resources, fields
from import_export.widgets import ForeignKeyWidget
from masters.models import MstSubCategoryTwo, MstSubCategory
from .utility import before_save_instance_update_create_date, CustomForeignKeyWidget


class MstSubCategoryTwoResource(resources.ModelResource):    
    sub_category_id = fields.Field(
        column_name='sub_category_id',
        attribute='sub_category_id',
        widget=ForeignKeyWidget(MstSubCategory, field='id')
    )

    class Meta:
        model = MstSubCategoryTwo
        fields = ('id', 'sub_2_category_name', 'sub_category_id')
        import_id_fields = ('sub_2_category_name', 'sub_category_id') 
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "SubCategoryTwos"
        update_on_import = True
    
    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):        
        if instance:
            if not instance.created_at:                
                instance = before_save_instance_update_create_date(instance, MstSubCategoryTwo)
