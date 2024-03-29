# Generated by Django 2.1.5 on 2019-05-13 11:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0030_auto_20190501_1853'),
    ]

    operations = [
        migrations.CreateModel(
            name='Consultation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symptomes', models.CharField(blank=True, max_length=2000, null=True)),
                ('resultat', models.CharField(blank=True, max_length=2000, null=True)),
                ('date', models.CharField(blank=True, max_length=2000, null=True)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='consultations', to='main.Patient')),
            ],
        ),
    ]
