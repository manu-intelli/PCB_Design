from import_export import resources
from masters.models import MstComponent
from .utility import before_save_instance_update_create_date


class MstComponentResource(resources.ModelResource):    
    class Meta:
        model = MstComponent
        fields = ('id', 'component_name', 'description')
        import_id_fields = ('component_name',)
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "Components"
        update_on_import = True
    
    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):        
        if instance:
            if not instance.created_at:
                instance = before_save_instance_update_create_date(instance, MstComponent)
