# Generated by Django 2.1.5 on 2019-04-11 17:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0017_auto_20190411_1828'),
    ]

    operations = [
        migrations.AlterField(
            model_name='medecin',
            name='csv_file',
            field=models.FileField(blank=True, null=True, upload_to='main/static/main/csv/medecins'),
        ),
    ]
