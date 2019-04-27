"""pcd URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views
#from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings
from django.conf.urls.static import static

app_name = "main"
urlpatterns = [
    path("", views.homepage, name="homepage"),
    path("qui-etes-vous", views.register, name="register"),
    path("inscription_medecin", views.inscription_medecin, name="inscription_medecin"),
    path("login_medecin", views.login_medecin, name="login_medecin"),
    path("acceuil", views.acceuil_medecin, name="acceuil_medecin"),
    path("profile", views.profile, name="profile"),
    path("logout", views.logout_request, name="logout"),
    path("affichage_fichier/<int:id>", views.affichage_fichier, name="affichage_fichier"),
    path("mes_patients", views.liste_des_patients, name="liste_des_patients"),
    path("mes_fichiers", views.liste_des_fichiers, name="liste_des_fichiers"),
    path("predictions", views.predictions, name="predictions"),
    path("stats", views.stats, name="stats"),
    path("ajout_fichier", views.ajout_fichier, name="ajout_fichier"),
    path("ajout_patient", views.ajout_patient, name="ajout_patient"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
