# Generated by Django 2.1.5 on 2019-05-01 17:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0029_auto_20190501_1850'),
    ]

    operations = [
        migrations.AlterField(
            model_name='medecin',
            name='patients',
            field=models.ManyToManyField(blank=True, null=True, related_name='medecins', to='main.Patient'),
        ),
    ]
