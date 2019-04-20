from django import forms
from django.contrib.auth.models import User
from django.utils.translation import gettext as _

from main.models import Medecin


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
        fields = ('date_de_naissance', 'genre')
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

