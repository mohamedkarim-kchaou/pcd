from django.contrib import messages
from django.contrib.auth import logout, authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
import csv
import ntpath
import datetime
import os
from main.models import Medecin, Patient, Partenaire, FichierCsv
from .forms import FormMedecin, FormUser, FormMedecinPhoto, FormAnnee, FormPrediction, FormRegion, FormCsv, FormCsvCreation
from .functions import afficher_annees_precedentes, predire_lstm1, predire_lstm4, predire_machine_learning, predire_reseaux_des_neurones, predire_region
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
                        first_name=instance.first_name, last_name=instance.last_name, region=instance.region)
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
        print(annee)
        csv = "flu"+annee
        lisre_regions_epidemiologique = afficher_annees_precedentes(settings.MEDIA_URL+'csv/'+csv+'.csv', annee)
        for element in os.listdir(settings.STATIC_URL+'main/images/resultats_annees_precedentes/'+annee):
            liste.append('main/images/resultats_annees_precedentes/'+annee+'/'+element)
        return render(request,
                      "main/acceuil_medecin.html",
                      {"medecin": medecin,
                       "form_annee": form_annee,
                       "liste": liste,
                       "liste_regions": lisre_regions_epidemiologique})

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


def predictions(request):
    medecin = Medecin.objects.get(user=request.user)
    liste = []
    erreur = []
    if request.method == 'POST':
        print("hey1")
        form_prediction = FormPrediction(request.POST)
        form_region = FormRegion(request.POST)
        if form_prediction.is_valid():
            print("hey")
            algorithme = "lstm1"
            """if algorithme == "lstm1":
                erreur = predire_lstm1()
            if algorithme == "lstm4":
                erreur = predire_lstm4()
            if algorithme == "reseaux des neurones":
                erreur = predire_reseaux_des_neurones()
            if algorithme == "machine learning":
                erreur = predire_machine_learning()"""
            for element in os.listdir(settings.STATIC_URL + 'main/images/predictions/'+algorithme):
                liste.append('main/images/predictions/'+algorithme+'/' + element)
            erreur = predire_lstm1()
        if form_region.is_valid():
            region = form_region['region'].value()
            print("heeeeeey")
            erreur = predire_region(region)
            for element in os.listdir(settings.STATIC_URL + 'main/images/predictions/regions/'+region):
                liste.append('main/images/predictions/regions/'+region+'/' + element)
        return render(request,
                      "main/predections.html",
                      {"medecin": medecin,
                       "erreur": erreur,
                       "liste": liste,
                       "form_prediction": form_prediction,
                       "form_region": form_region})
    form_prediction = FormPrediction
    form_region = FormRegion
    return render(request,
                  "main/predections.html",
                  {"medecin": medecin,
                   "liste": liste,
                   "form_prediction": form_prediction,
                   "form_region": form_region})


def stats(request):
    medecin = Medecin.objects.get(user=request.user)
    liste = []
    liste1 = []
    liste2 = []
    liste3 = []
    liste4 = []
    if request.method == 'POST':
        form_annee = FormAnnee(request.POST)
        if form_annee.is_valid():
            annee = form_annee['annee'].value()
            categorie = form_annee['categorie'].value()
            option1 = 0
            option2 = 0
            option3 = 0
            option4 = 0
            print(annee)
            csv = "flu" + annee
            liste_regions_epidemiologique = afficher_annees_precedentes(settings.MEDIA_URL + 'csv/' + csv + '.csv', annee)
            for element in os.listdir(settings.STATIC_URL + 'main/images/resultats_annees_precedentes/' + annee):
                liste.append('main/images/resultats_annees_precedentes/' + annee + '/' + element)
            if categorie == "Nombre total des cas":
                option1 = 1
                liste1 = [liste[0], liste[1], liste[2]]
            if categorie == "Nombre total des cas de grippe par tranches d'âge":
                option2 = 1
                liste2 = [liste[3], liste[4], liste[5]]
            if categorie == "Nombre total des cas SARI par tranches d'âge":
                option3 = 1
                liste3 = [liste[6], liste[7], liste[8]]
            if categorie == "Le pourcentage des cas de grippe et SARI":
                option4 = 1
                liste4 = [liste[9], liste[10]]
            return render(request,
                          "main/stats.html",
                          {"medecin": medecin,
                           "form_annee": form_annee,
                           "liste": liste,
                           "annee": annee,
                           "option1": option1,
                           "option2": option2,
                           "option3": option3,
                           "option4": option4,
                           "liste1": liste1,
                           "liste2": liste2,
                           "liste3": liste3,
                           "liste4": liste4,
                           "liste_regions": liste_regions_epidemiologique})
    form_annee = FormAnnee
    return render(request,
                  "main/stats.html",
                  {"medecin": medecin,
                   "liste": liste,
                   "form_annee": form_annee})


def ajout_fichier(request):
    medecin = request.user.medecin

    form_csv_creation = FormCsvCreation
    if request.method == 'POST':
        form_csv_creation = FormCsvCreation(request.POST)
        if form_csv_creation.is_valid():
            nom = len(request.user.medecin.csv_files.all())+1
            username = request.user.username
            region = request.user.medecin.region
            grippe_enfant = form_csv_creation.cleaned_data.get('grippe_enfant')
            sari_enfant = form_csv_creation.cleaned_data.get('sari_enfant')
            consultation_enfant = form_csv_creation.cleaned_data.get('consultation_enfant')
            grippe_adolescent = form_csv_creation.cleaned_data.get('grippe_adolescent')
            sari_adolescent = form_csv_creation.cleaned_data.get('sari_adolescent')
            consultation_adolescent = form_csv_creation.cleaned_data.get('consultation_adolescent')
            grippe_mur = form_csv_creation.cleaned_data.get('grippe_mur')
            sari_mur = form_csv_creation.cleaned_data.get('sari_mur')
            consultation_mur = form_csv_creation.cleaned_data.get('consultation_mur')

            tot_grippe = grippe_adolescent + grippe_enfant + grippe_mur
            tot_sari = sari_adolescent + sari_enfant + sari_mur
            tot_consultation = consultation_adolescent + consultation_enfant + consultation_mur
            if tot_consultation:
                taux_grippe = tot_grippe / tot_consultation
                taux_sari = tot_sari / tot_consultation
            else:
                taux_grippe = 0
                taux_sari = 0

            now = datetime.datetime.now()
            my_date = datetime.date.today()
            year, week_num, day_of_week = my_date.isocalendar()
            mois = ['', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                    'october', 'november', 'december']

            mylist = [region,
                      grippe_enfant, sari_enfant, consultation_enfant,
                      grippe_adolescent, sari_adolescent, consultation_adolescent,
                      grippe_mur, sari_mur, consultation_mur,
                      tot_grippe, tot_sari, tot_consultation,
                      week_num, mois[now.month], year,
                      taux_grippe, taux_sari]
            print("hey")
            with open(settings.MEDIA_URL + 'csv/medecins/' + str(username)+"_"+str(nom) + '.csv', 'w') as myfile:
                wr = csv.writer(myfile, delimiter=',')
                wr.writerow(mylist)
            csv1 = FichierCsv(medecin=request.user.medecin, csv_file='csv/medecins/' + str(username)+"_"+str(nom) + '.csv')
            csv1.save()
            tf = open(settings.MEDIA_URL +'csv/tot.csv', 'a')
            tf.write(open(settings.MEDIA_URL +'csv/medecins/' + str(username)+"_"+str(nom) + '.csv').read())
            tf.close()
            return redirect("main:acceuil_medecin")
    return render(request,
                  "main/ajout_fichier.html",
                  {"medecin": medecin,
                   "form_csv_creation": form_csv_creation})


def ajout_patient():
    return render()
