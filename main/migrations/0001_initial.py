# Generated by Django 2.1.5 on 2019-04-02 23:27

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Medecin',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nom_medecin', models.CharField(default='nom', max_length=200)),
                ('prenom_medecin', models.CharField(default='prenom', max_length=200)),
                ('username_medecin', models.CharField(default='username', max_length=200)),
                ('email_medecin', models.EmailField(default='medecin@gmail.com', max_length=254)),
            ],
        ),
        migrations.CreateModel(
            name='Tu',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Tu_title', models.CharField(max_length=200)),
            ],
        ),
    ]
