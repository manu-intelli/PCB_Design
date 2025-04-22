from import_export import resources, fields
from masters.models import MstSectionRules, MstDesignOptions, MstSectionGroupings
from .utility import before_save_instance_update_create_date, CustomManyToManyWidget



class MstSectionGroupingsResource(resources.ModelResource):
    rules = fields.Field(
        column_name='rules',
        attribute='rules',        
        widget=CustomManyToManyWidget(MstSectionRules, field='id', separator=',')
    )
    design_options = fields.Field(
        column_name='design_options',
        attribute='design_options',
        widget=CustomManyToManyWidget(MstDesignOptions, field='id',  separator=',')
    )

    class Meta:
        model = MstSectionGroupings
        fields = ('id', 'design_doc', 'section_name', 'rules', 'design_options')
        import_id_fields = ('design_doc', 'section_name')
        primary_key = 'id'
        skip_unchanged = True
        report_skipped = False
        sheet_name = "SectionGroupings"
        update_on_import = True

    def before_save_instance(self, instance, row, using_transactions, dry_run, **kwargs):
        """
        This method is used to modify instances before they are saved to the database.
        You can add any custom logic here, such as setting timestamps or modifying fields before saving.
        """
        if instance:
            if not instance.created_at:
                instance = before_save_instance_update_create_date(instance, MstSectionGroupings)
