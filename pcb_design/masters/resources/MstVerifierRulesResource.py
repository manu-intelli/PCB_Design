from import_export import resources, fields
from masters.models import MstVerifierRules, MstVerifierField
from .utility import before_save_instance_update_create_date, CustomForeignKeyWidget
from import_export.widgets import ForeignKeyWidget


class MstVerifierRulesResource(resources.ModelResource):    
    verifier_field = fields.Field( column_name='verifier_field', attribute='verifier_field',
                             widget=ForeignKeyWidget(MstVerifierField, field='id'))

    class Meta:
        model = MstVerifierRules
        fields = ('id', 'verifier_field', 'design_doc', 'rule_number', 'conditional_var', 'value')
        import_id_fields = ('id',)
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "VerifierRules"
        update_on_import = True

    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):
        """Override before_save_instance to update the created_at field."""
        if instance:
            if not instance.created_at:
                instance = before_save_instance_update_create_date(instance, MstVerifierRules)
