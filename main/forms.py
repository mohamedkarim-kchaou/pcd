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
        widget=forms.PasswordInput,
        strip=False,
        help_text=_("Enter the same password as before, for verification."),
    )

    class Meta:
        model = User
        fields = ('username', 'password', 'email', 'first_name', 'last_name')
        widgets = {
            'password': forms.PasswordInput()
        }


class FormMedecin(forms.ModelForm):
    class Meta:
        model = Medecin
        fields = ('sexe',)


