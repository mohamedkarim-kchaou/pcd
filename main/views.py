from django.contrib import messages
from django.contrib.auth import logout, authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
import csv,os


from main.models import Medecin
from .forms import FormMedecin, FormUser, FormMedecinCsv


# Create your views here.
def homepage(request):
    return render(request=request,
                  template_name="main/home.html", )


def register(request):
    return render(request=request,
                  template_name="main/register.html")


def inscription_medecin(request):
    if request.method == "POST":
        print("hey")
        form_user = FormUser(request.POST)
        form_medecin = FormMedecin(request.POST)
        if form_user.is_valid():
            print("hey")
            instance = form_user.instance
            user = User(username=instance.username, password=make_password(instance.password))
            user.save()
            login(request, user)
            if form_medecin.is_valid():
                print("hey")
                medecin = form_medecin.instance
                medecin.user = user
                medecin.save()
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
    form_medecin_csv = FormMedecinCsv(request.POST, request.FILES)
    if request.method == 'POST':
        if form_medecin_csv.is_valid():
            medecin = Medecin.objects.get(user=request.user)
            csv_file = form_medecin_csv.cleaned_data.get('csv_file')
            medecin.csv_file = csv_file
            medecin.save()
            return redirect("main:acceuil_medecin")
    return render(request,
                  "main/acceuil_medecin.html",
                  {"patients": request.user.medecin.patients.all(), "form": form_medecin_csv})


def profile(request):
    return render(request=request,
                  template_name="main/profile.html")


def logout_request(request):
    logout(request)
    #messages.info(request, "Logged out successfully!")
    return redirect("main:homepage")


def lire_fichier(request):
    fichier = request.user.medecin.csv_file
    f = os.path.basename(fichier.name)
    regions = [[]]
    with open(str(fichier)) as csvfile:
        contenu = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in contenu:
            regions.append(f)
    return render(request=request,
                  template_name="main/lire_fichier.html",
                  context={"contenu": regions})
