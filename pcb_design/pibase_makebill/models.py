from django.db import models
from django.conf import settings
from pibase.models import PiBaseRecord, PiBaseComponent  # adjust import as needed

class MakeBillRecord(models.Model):
    # pi_base_record = models.ForeignKey(PiBaseRecord, on_delete=models.CASCADE, related_name="make_bills")

    op_number = models.CharField(max_length=20, verbose_name="OP Number")
    opu_number = models.CharField(max_length=20, verbose_name="OPU Number")
    edu_number = models.CharField(max_length=20, verbose_name="EDU Number")
    model_name = models.CharField(max_length=100)

    # Store selected components (13 only)
    components = models.ManyToManyField(PiBaseComponent, blank=True, related_name="make_bill_records", verbose_name="Selected Components")

    # Copy JSON fields from PiBaseRecord
    case_style_data = models.JSONField(default=dict, verbose_name="Case Style Data")

    can_details = models.JSONField(default=dict, blank=True, null=True)
    pcb_details = models.JSONField(default=dict, blank=True, null=True)
    chip_aircoil_details = models.JSONField(default=dict, blank=True, null=True)
    chip_inductor_details = models.JSONField(default=dict, blank=True, null=True)
    chip_capacitor_details = models.JSONField(default=dict, blank=True, null=True)
    chip_resistor_details = models.JSONField(default=dict, blank=True, null=True)
    transformer_details = models.JSONField(default=dict, blank=True, null=True)
    shield_details = models.JSONField(default=dict, blank=True, null=True)
    finger_details = models.JSONField(default=dict, blank=True, null=True)
    copper_flaps_details = models.JSONField(default=dict, blank=True, null=True)
    resonator_details = models.JSONField(default=dict, blank=True, null=True)
    ltcc_details = models.JSONField(default=dict, blank=True, null=True)
    special_requirements = models.JSONField(default=dict, blank=True, null=True)

    # User and timestamps
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='make_bill_created_by')
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='make_bill_updated_by')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MakeBill for {self.pi_base_record.model_name} (OP#{self.pi_base_record.op_number})"

    class Meta:
        db_table = 'MakeBillRecord'
        verbose_name = "Make Bill Record"
        verbose_name_plural = "Make Bill Records"
        ordering = ['-created_at']
