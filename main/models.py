from django.contrib.auth.models import User
from django.db import models

from django.db.models import CASCADE


class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)
    sexe = models.BooleanField(default=0)

    def __str__(self):
        return str(self.user)


class Medecin(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)
    CHOIXGENRE = (('HOMME', 'HOMME'), ('FEMME', 'FEMME'))
    genre = models.CharField(max_length=10, choices=CHOIXGENRE, default='HOMME',)
    csv_file = models.FileField(null=True, blank=True, upload_to='main/static/main/csv/')
    date_de_naissance = models.DateField(null=True, blank=True)
    patients = models.ManyToManyField(Patient, related_name="medecins")

    def __str__(self):
        return str(self.user)


class Partenaire(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)

    def __str__(self):
        return str(self.user)
