from import_export import resources
from masters.models import MstSectionRules
from .utility import before_save_instance_update_create_date


class MstSectionRulesResource(resources.ModelResource):
    class Meta:
        model = MstSectionRules
        fields = ('id', 'design_doc', 'rule_number', 'parameter', 'min_value', 'max_value', 'nominal', 'comments')
        import_id_fields = ('rule_number', 'design_doc')
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "SectionRules"
        update_on_import = True

    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):
        if instance:            
            if not instance.created_at:
                instance = before_save_instance_update_create_date(instance, MstSectionRules)