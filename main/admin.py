from django.contrib import admin

# Register your models here.
from main.models import Medecin, Patient, Partenaire, FichierCsv, Consultation


class MedecinAdmin(admin.ModelAdmin):
    fieldsets = [
        ("title", {"fields": ["username"]})
    ]


admin.site.register(Medecin)
admin.site.register(Patient)
admin.site.register(Partenaire)
admin.site.register(FichierCsv)
admin.site.register(Consultation)
