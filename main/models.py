from django.contrib.auth.models import User
from django.db import models
from django.db.models import CASCADE


class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)
    sexe = models.BooleanField(default=0)
    photo = models.ImageField(upload_to='images/patients', default='images/medecin_login.png')

    def __str__(self):
        return str(self.user)


class Medecin(models.Model):
    objects = models.Manager()  # Ignore IDE warning
    user = models.OneToOneField(User, on_delete=CASCADE)
    CHOIXGENRE = (('HOMME', 'HOMME'), ('FEMME', 'FEMME'))
    genre = models.CharField(max_length=10, choices=CHOIXGENRE, default='HOMME',)
    date_de_naissance = models.DateField(null=True, blank=True)
    photo = models.ImageField(upload_to='images/medecins', default='images/medecin_login.png')
    patients = models.ManyToManyField(Patient, related_name="medecins")

    def __str__(self):
        return str(self.user)


class Partenaire(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)

    def __str__(self):
        return str(self.user)


class FichierCsv(models.Model):
    medecin = models.ForeignKey('Medecin', on_delete=CASCADE, related_name="csv_files")
    csv_file = models.FileField(null=True, blank=True, upload_to='csv/medecins')

    def __str__(self):
        return str(self.csv_file)
