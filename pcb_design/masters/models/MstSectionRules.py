from django.db import models

from .BaseModel import BaseModel
from authentication.alias import AliasField


class MstSectionRules(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    design_doc = models.CharField(max_length=255, db_column='DESIGN_DOC')
    rule_number = models.CharField(max_length=50, db_column='RULE_NUMBER')
    parameter = models.CharField(max_length=1255, db_column='PARAMETER')
    min_value = models.CharField(max_length=10, blank=True, null=True, db_column='MIN_VALUE')
    max_value = models.CharField(max_length=10, blank=True, null=True, db_column='MAX_VALUE')
    nominal = models.CharField(max_length=10, blank=True, null=True, db_column='NOMINAL')
    comments = models.TextField(blank=True, null=True, db_column='COMMENTS')
    name = AliasField(db_column='RULE_NUMBER', blank=True, null=True, editable=False)

    class Meta:
        unique_together = ('design_doc', 'rule_number')
        verbose_name = '06 Section Rule'
        verbose_name_plural = '06 Section Rules'

    def __str__(self):
        return f"Rule {self.rule_number}"
