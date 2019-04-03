from django.contrib.auth.models import User
from django.db import models
import datetime

from django.db.models import CASCADE
from django.utils.translation import gettext as _


class Medecin(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)
    sexe = models.BooleanField(default=0, null=True, blank=True)

    def __str__(self):
        return str(self.user)


class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)
    sexe = models.BooleanField(default=0)

    def __str__(self):
        return str(self.user)


class Partenaire(models.Model):
    user = models.OneToOneField(User, on_delete=CASCADE)

    def __str__(self):
        return str(self.user)
