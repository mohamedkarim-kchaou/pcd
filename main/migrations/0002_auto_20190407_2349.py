# Generated by Django 2.1.5 on 2019-04-07 22:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='medecin',
            old_name='sexe',
            new_name='genre',
        ),
    ]
