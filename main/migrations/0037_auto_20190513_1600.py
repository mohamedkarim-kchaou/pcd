# Generated by Django 2.1.5 on 2019-05-13 15:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0036_auto_20190513_1545'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='consultation',
            name='symptomes',
        ),
        migrations.AddField(
            model_name='consultation',
            name='symptomes',
            field=models.CharField(blank=True, max_length=2000, null=True),
        ),
    ]
