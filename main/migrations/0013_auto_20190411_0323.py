# Generated by Django 2.1.5 on 2019-04-11 02:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0012_auto_20190411_0307'),
    ]

    operations = [
        migrations.AlterField(
            model_name='medecin',
            name='photo',
            field=models.ImageField(default='images/medecins/medecin_login.png', upload_to='images/medecins'),
        ),
    ]
