from django.contrib import messages
from django.contrib.auth import logout, authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
import csv
import ntpath
import os
from main.models import Medecin, Patient
from .forms import FormMedecin, FormUser, FormMedecinPhoto, FormAnnee
from .functions import afficher_annees_precedentes
from django.conf import settings


# Create your views here.
def homepage(request):
    return render(request=request,
                  template_name="main/home.html", )


def register(request):
    return render(request=request,
                  template_name="main/register.html")


def inscription_medecin(request):
    if request.method == "POST":
        form_user = FormUser(request.POST)
        form_medecin = FormMedecin(request.POST)
        if form_user.is_valid():
            instance = form_user.instance
            user = User(username=instance.username, password=make_password(instance.password), email=instance.email,
                        first_name=instance.first_name, last_name=instance.last_name)
            user.save()
            if form_medecin.is_valid():
                medecin = form_medecin.instance
                medecin.user = user
                medecin.save()
                login(request, user)
            return redirect("main:acceuil_medecin")

        else:
            for msg in form_user.error_messages:
                messages.error(request, f"{msg}: {form_user.error_messages[msg]}")

            return render(request=request,
                          template_name="main/inscription_medecin.html",
                          context={"form_user": form_user, "form_medecin": form_medecin})
    else:
        form_user = FormUser  # UserCreationForm
        form_medecin = FormMedecin
        return render(request=request,
                      template_name="main/inscription_medecin.html",
                      context={"form_user": form_user, "form_medecin": form_medecin})


def login_medecin(request):
    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}")
                return redirect("main:acceuil_medecin")
            else:
                print("problem")
                messages.error(request, "Invalid username or password.")
        else:
            print("error")
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            print(username, password)
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request=request,
                  template_name="main/login_medecin.html",
                  context={"login_form": form})


def acceuil_medecin(request):
    medecin = Medecin.objects.get(user=request.user)
    liste = []
    """verification = []
    for element in os.listdir(settings.STATIC_URL + 'main/images/resultats_annees_precedentes' + '/09-10'):
        verification.append(element)"""
    if request.method == 'POST':
        form_annee = FormAnnee(request.POST)
        annee = form_annee['annee'].value()
        csv = "flu"+annee
        afficher_annees_precedentes(settings.MEDIA_URL+'csv/'+csv+'.csv', annee)
        for element in os.listdir(settings.STATIC_URL+'main/images/resultats_annees_precedentes/'+annee):
            liste.append('main/images/resultats_annees_precedentes/'+annee+'/'+element)
        return render(request,
                      "main/acceuil_medecin.html",
                      {"medecin": medecin,
                       "form_annee": form_annee,
                       "liste": liste})
    #form_medecin_csv = FormMedecinCsv(request.POST, request.FILES)
    """form_medecin_photo = FormMedecinPhoto(request.POST, request.FILES)
    if request.method == 'POST':
        """"""if form_medecin_csv.is_valid():
            csv_file = form_medecin_csv.cleaned_data.get('csv_file')
            medecin.csv_file = csv_file
            medecin.save()""""""
        if form_medecin_photo.is_valid():
            photo = form_medecin_photo.cleaned_data.get('photo')
            medecin.photo = photo
            medecin.save()
        return redirect("main:acceuil_medecin")"""
    form_annee = FormAnnee
    return render(request,
                  "main/acceuil_medecin.html",
                  {"medecin": medecin,
                   "liste": liste,
                   "form_annee": form_annee})
                   #"csv_file_name": csv_file_name,
                   #"form_medecin_csv": form_medecin_csv,
                   #"form_medecin_photo": form_medecin_photo})


def profile(request):
    return render(request=request,
                  template_name="main/profile.html")


def liste_des_patients(request):
    return render(request=request,
                  template_name="main/listes_des_patients.html",
                  context={"patients": request.user.medecin.patients.all(),
                           "medecin": request.user.medecin})


def liste_des_fichiers(request):
    csv_files = request.user.medecin.csv_files.all()
    liste = []
    if csv_files:
        for fichier in csv_files:
            nom_fichier = fichier.csv_file.name.split('/')
            liste.append(nom_fichier[-1])
    else:
        csv_file_name = "You haven't uploaded any csv files yet."
    for i in liste:
        print(i)
    return render(request=request,
                  template_name="main/listes_des_fichiers.html",
                  context={"fichiers": request.user.medecin.csv_files.all(),
                           "medecin": request.user.medecin,
                           "liste_fichiers": liste})


def logout_request(request):
    logout(request)
    #messages.info(request, "Logged out successfully!")
    return redirect("main:homepage")


def affichage_fichier(request, id):
    fichier = request.user.medecin.csv_files.all()[id].csv_file
    regions = [[]]
    with open(fichier.url, 'r') as csvfile:
        contenu = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in contenu:
            regions.append(row)
    return render(request=request,
                  template_name="main/affichage_fichier.html",
                  context={"contenu": regions})
