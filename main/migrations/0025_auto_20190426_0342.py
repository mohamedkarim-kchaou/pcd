# Generated by Django 2.1.5 on 2019-04-26 02:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0024_auto_20190412_2025'),
    ]

    operations = [
        migrations.AlterField(
            model_name='patient',
            name='photo',
            field=models.ImageField(default='images/patient.png', upload_to='images/patients'),
        ),
    ]
