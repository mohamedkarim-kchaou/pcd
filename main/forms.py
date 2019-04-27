from django import forms
from django.contrib.auth.models import User
from django.utils.translation import gettext as _

from main.models import Medecin, FichierCsv, Patient, Partenaire


class FormUser(forms.ModelForm):
    error_messages = {
        'password_mismatch': _("The two password fields didn't match."),
    }
    password2 = forms.CharField(
        label=_("Password confirmation"),
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        strip=False,
        help_text=_("Enter the same password as before, for verification."),
    )

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'username', 'password', 'email')
        widgets = {
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'password': forms.PasswordInput(attrs={'class': 'form-control'})
        }


class FormMedecin(forms.ModelForm):
    class Meta:
        model = Medecin
        fields = ('date_de_naissance', 'genre', 'region')
        widget = {
            'genre': forms.Select,
        }
    date_de_naissance = forms.DateField(
            widget=forms.DateInput(attrs={'class': 'datepicker'}, format='%b %d, %Y'),
            input_formats=('%b %d, %Y',)
    )

"""
class FormMedecinCsv(forms.ModelForm):
    class Meta:
        model = Medecin
        fields = ('csv_file',)"""


class FormMedecinPhoto(forms.ModelForm):
    class Meta:
        model = Medecin
        fields = ('photo',)


class FormAnnee(forms.Form):
    annee = forms.ChoiceField(choices=[('09-10', '09-10'), ('10-11', '10-11'), ('11-12', '11-12'), ('12-13', '12-13'),
                                       ('14-15', '14-15'), ('15-16', '15-16'), ('16-17', '16-17'), ('17-18', '17-18')])
    categorie = forms.ChoiceField(choices=[('Nombre total des cas', 'Nombre total des cas'),
                                           ("Nombre total des cas de grippe par tranches d'âge",
                                            "Nombre total des cas de grippe par tranches d'âge"),
                                           ("Nombre total des cas SARI par tranches d'âge",
                                            "Nombre total des cas SARI par tranches d'âge"),
                                           ("Le pourcentage des cas de grippe et SARI",
                                            "Le pourcentage des cas de grippe et SARI")])


class FormPrediction(forms.Form):
    """algorithme = forms.ChoiceField(choices=[('lstm1', 'lstm1'), ('lstm4', 'lstm4'),
                                            ('machine learning', 'machine learning'),
                                            ('reseaux des neurones', 'reseaux des neurones')])"""


class FormRegion(forms.Form):
    region = forms.ChoiceField(choices=[('Tunis', 'Tunis'), ('Ariana', 'Ariana'), ('Mannouba', 'Mannouba'),
                                        ('Ben Arous', 'Ben Arous'), ('Bizerte', 'Bizerte'), ('Nabeul', 'Nabeul'),
                                        ('Zaghouan', 'Zaghouan'), ('Beja', 'Beja'), ('Jendouba', 'Jendouba'),
                                        ('Le Kef', 'Le Kef'), ('Siliana', 'Siliana'), ('Kairouan', 'Kairouan'),
                                        ('Sousse', 'Sousse'), ('Mahdia', 'Mahdia'), ('Monastir', 'Monastir'),
                                        ('Kasserine', 'Kasserine'), ('Sfax', 'Sfax'), ('Gabes', 'Gabes'),
                                        ('Kebili', 'Kebili'), ('Gafsa', 'Gafsa'), ('Sidi Bouzid', 'Sidi Bouzid'),
                                        ('Tozeur', 'Tozeur'), ('Medenine', 'Medenine'), ('Tataouin', 'Tataouin')])


class FormCsv(forms.ModelForm):
    class Meta:
        model = FichierCsv
        fields = ('csv_file',)


class FormCsvCreation(forms.Form):
    """region = forms.ChoiceField(choices=[('Tunis', 'Tunis'), ('Ariana', 'Ariana'), ('Mannouba', 'Mannouba'),
                                    ('Ben Arous', 'Ben Arous'), ('Bizerte', 'Bizerte'), ('Nabeul', 'Nabeul'),
                                    ('Zaghouan', 'Zaghouan'), ('Beja', 'Beja'), ('Jendouba', 'Jendouba'),
                                    ('Le Kef', 'Le Kef'), ('Siliana', 'Siliana'), ('Kairouan', 'Kairouan'),
                                    ('Sousse', 'Sousse'), ('Mahdia', 'Mahdia'), ('Monastir', 'Monastir'),
                                    ('Kasserine', 'Kasserine'), ('Sfax', 'Sfax'), ('Gabes', 'Gabes'),
                                    ('Kebili', 'Kebili'), ('Gafsa', 'Gafsa'), ('Sidi Bouzid', 'Sidi Bouzid'),
                                    ('Tozeur', 'Tozeur'), ('Medenine', 'Medenine'), ('Tataouin', 'Tataouin')])
    mois = forms.ChoiceField(choices=[('january', 'Janvier'), ('february', 'Fevrier'), ('march', 'Mars'),
                                      ('april', 'Avril'), ('may', 'Mai'), ('june', 'Juin'),
                                      ('july', 'Juillet'), ('august', 'Août'), ('september', 'Septembre'),
                                      ('october', 'Octobre'), ('november', 'Novembre'), ('december', 'Decembre')])"""
    grippe_enfant = forms.IntegerField(initial=0)
    sari_enfant = forms.IntegerField(initial=0)
    consultation_enfant = forms.IntegerField(initial=0)
    grippe_adolescent = forms.IntegerField(initial=0)
    sari_adolescent = forms.IntegerField(initial=0)
    consultation_adolescent = forms.IntegerField(initial=0)
    grippe_mur = forms.IntegerField(initial=0)
    sari_mur = forms.IntegerField(initial=0)
    consultation_mur = forms.IntegerField(initial=0)

