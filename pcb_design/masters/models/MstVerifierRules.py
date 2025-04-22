from django.db import models

from .MstVerifierField import MstVerifierField
from .BaseModel import BaseModel
from authentication.alias import AliasField


class MstVerifierRules(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    verifier_field = models.ForeignKey(MstVerifierField, on_delete=models.CASCADE, db_column='VERIFIER_FIELD_ID')
    design_doc = models.CharField(max_length=255, db_column='DESIGN_DOC')
    rule_number = models.CharField(max_length=50, db_column='RULE_NUMBER')
    conditional_var = models.CharField(max_length=50, db_column='CONDITIONAL_VAR', blank=True, null=True)
    value = models.CharField(max_length=10, db_column='VALUE', blank=True, null=True)
    name = AliasField(db_column='RULE_NUMBER', blank=True, null=True, editable=False)

    class Meta:        
        verbose_name = '09 Verifier Rule'
        verbose_name_plural = '09 Verifier Rules'
        unique_together = ('verifier_field', 'design_doc', 'rule_number')

    def __str__(self):
        return f"Rule {self.rule_number} for {self.verifier_field.field_name}"
