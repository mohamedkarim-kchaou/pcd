from django.contrib import admin

# Register your models here.
from main.models import Medecin, Patient, Partenaire


class MedecinAdmin(admin.ModelAdmin):
    fieldsets = [
        ("title", {"fields": ["username"]})
    ]


admin.site.register(Medecin)
admin.site.register(Patient)
admin.site.register(Partenaire)