# Generated by Django 2.1.5 on 2019-04-10 01:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0006_auto_20190410_0158'),
    ]

    operations = [
        migrations.AlterField(
            model_name='medecin',
            name='photo',
            field=models.ImageField(default='images/medecin_login.png', upload_to='images/'),
        ),
    ]
