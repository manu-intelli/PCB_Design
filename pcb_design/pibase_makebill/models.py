from django.db import models
from django.conf import settings


class MakeBillStatus(models.Model):
    """
    Stores the status for MakeBillRecord.
    """
    STATUS_CHOICES = (
        (1, 'Pending'),
        (2, 'Complete'),
        (3, 'Close'),
        (4, 'Delete'),
    )

    status_code = models.PositiveSmallIntegerField(choices=STATUS_CHOICES, unique=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return dict(self.STATUS_CHOICES).get(self.status_code, "Unknown")

    class Meta:
        db_table = 'MakeBillStatus'
        verbose_name = "Make Bill Status"
        verbose_name_plural = "Make Bill Statuses"
        ordering = ['status_code']

class MakeBillRecord(models.Model):
    """
    Stores Make Bill records, including selected components and copied JSON fields from PiBaseRecord.
    """
    op_number = models.CharField(max_length=20, verbose_name="OP Number")
    opu_number = models.CharField(max_length=20, verbose_name="OPU Number")
    edu_number = models.CharField(max_length=20, verbose_name="EDU Number")
    model_name = models.CharField(max_length=100)

    # Store selected components (13 only)
    components = models.JSONField(default=dict, blank=True, null=True)

    # Copy JSON fields from PiBaseRecord
    case_style_data = models.JSONField(default=dict, blank=True, null=True, verbose_name="Case Style Data")
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

    revision_number = models.CharField(max_length=20)
    pibaseRecord = models.JSONField(default=dict, blank=True, null=True)

    status = models.ForeignKey(
        MakeBillStatus,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="make_bill_records",
        verbose_name="Status"
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='make_bill_created_by'
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='make_bill_updated_by'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MakeBill for {self.model_name} (OP#{self.op_number})"

    class Meta:
        db_table = 'MakeBillRecord'
        verbose_name = "Make Bill Record"
        verbose_name_plural = "Make Bill Records"
        ordering = ['-created_at']
