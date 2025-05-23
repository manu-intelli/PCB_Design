# Generated by Django 5.0.10 on 2025-03-17 10:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('masters', '0011_alter_mstcategory_options_alter_mstcomponent_options_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mstconditions',
            name='comparison_max_value',
            field=models.DecimalField(blank=True, db_column='COMPARISON_MAX_VALUE', decimal_places=4, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='mstconditions',
            name='comparison_min_value',
            field=models.DecimalField(blank=True, db_column='COMPARISON_MIN_VALUE', decimal_places=4, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='mstconditions',
            name='comparison_operator',
            field=models.CharField(blank=True, choices=[('gte', 'Greater than or equal to'), ('gt', 'Greater than'), ('eq', 'Equal to'), ('lt', 'Less than'), ('lte', 'Less than or equal to'), ('range', 'Within a range'), ('add', 'Add'), ('sub', 'Subtract')], db_column='COMPARISON_OPERATOR', max_length=10, null=True),
        ),
        migrations.AlterField(
            model_name='mstconditions',
            name='condition_max_value',
            field=models.DecimalField(blank=True, db_column='CONDITION_MAX_VALUE', decimal_places=4, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='mstconditions',
            name='condition_min_value',
            field=models.DecimalField(blank=True, db_column='CONDITION_MIN_VALUE', decimal_places=4, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='mstconditions',
            name='condition_operator',
            field=models.CharField(blank=True, choices=[('gte', 'Greater than or equal to'), ('gt', 'Greater than'), ('eq', 'Equal to'), ('lt', 'Less than'), ('lte', 'Less than or equal to'), ('range', 'Within a range'), ('add', 'Add'), ('sub', 'Subtract')], db_column='CONDITION_OPERATOR', max_length=10, null=True),
        ),
    ]
