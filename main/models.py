from django.contrib.auth.models import User
from django.db import models
from django.db.models import CASCADE


class Patient(models.Model):
    objects = models.Manager()  # Ignore IDE warning
    user = models.OneToOneField(User, on_delete=CASCADE)
    photo = models.ImageField(upload_to='images/patients', default='images/patient.png')
    CHOIXGENRE = (('HOMME', 'HOMME'), ('FEMME', 'FEMME'))
    CHOIXREGION = [('Tunis', 'Tunis'), ('Ariana', 'Ariana'), ('Mannouba', 'Mannouba'),
                   ('Ben Arous', 'Ben Arous'), ('Bizerte', 'Bizerte'), ('Nabeul', 'Nabeul'),
                   ('Zaghouan', 'Zaghouan'), ('Beja', 'Beja'), ('Jendouba', 'Jendouba'),
                   ('Le Kef', 'Le Kef'), ('Siliana', 'Siliana'), ('Kairouan', 'Kairouan'),
                   ('Sousse', 'Sousse'), ('Mahdia', 'Mahdia'), ('Monastir', 'Monastir'),
                   ('Kasserine', 'Kasserine'), ('Sfax', 'Sfax'), ('Gabes', 'Gabes'),
                   ('Kebili', 'Kebili'), ('Gafsa', 'Gafsa'), ('Sidi Bouzid', 'Sidi Bouzid'),
                   ('Tozeur', 'Tozeur'), ('Medenine', 'Medenine'), ('Tataouin', 'Tataouin')]
    genre = models.CharField(max_length=10, choices=CHOIXGENRE, default='HOMME', )
    date_de_naissance = models.DateField(null=True, blank=True)
    region = models.CharField(max_length=20, choices=CHOIXREGION, default="Tunis")

    def __str__(self):
        return str(self.user)


class Medecin(models.Model):
    objects = models.Manager()  # Ignore IDE warning
    user = models.OneToOneField(User, on_delete=CASCADE)
    CHOIXGENRE = (('HOMME', 'HOMME'), ('FEMME', 'FEMME'))
    CHOIXREGION = [('Tunis', 'Tunis'), ('Ariana', 'Ariana'), ('Mannouba', 'Mannouba'),
                   ('Ben Arous', 'Ben Arous'), ('Bizerte', 'Bizerte'), ('Nabeul', 'Nabeul'),
                   ('Zaghouan', 'Zaghouan'), ('Beja', 'Beja'), ('Jendouba', 'Jendouba'),
                   ('Le Kef', 'Le Kef'), ('Siliana', 'Siliana'), ('Kairouan', 'Kairouan'),
                   ('Sousse', 'Sousse'), ('Mahdia', 'Mahdia'), ('Monastir', 'Monastir'),
                   ('Kasserine', 'Kasserine'), ('Sfax', 'Sfax'), ('Gabes', 'Gabes'),
                   ('Kebili', 'Kebili'), ('Gafsa', 'Gafsa'), ('Sidi Bouzid', 'Sidi Bouzid'),
                   ('Tozeur', 'Tozeur'), ('Medenine', 'Medenine'), ('Tataouin', 'Tataouin')]
    genre = models.CharField(max_length=10, choices=CHOIXGENRE, default='HOMME',)
    date_de_naissance = models.DateField(null=True, blank=True)
    photo = models.ImageField(upload_to='images/medecins', default='images/medecin_login.png')
    patients = models.ManyToManyField(Patient, related_name="medecins", null=True, blank=True)
    region = models.CharField(max_length=20, choices=CHOIXREGION, default="Tunis")
    first_connection = models.IntegerField(default=0)

    def __str__(self):
        return str(self.user)


class Partenaire(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)

    def __str__(self):
        return str(self.user)


class FichierCsv(models.Model):
    medecin = models.ForeignKey(Medecin, on_delete=CASCADE, related_name="csv_files")
    csv_file = models.FileField(null=True, blank=True, upload_to='csv/medecins')

    def __str__(self):
        return str(self.csv_file)
