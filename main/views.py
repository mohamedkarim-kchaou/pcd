from django.contrib import messages
from django.contrib.auth import logout, authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.shortcuts import render, redirect

from .forms import FormMedecin, FormUser


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
            instance = form_user.instance
            user = User(username=instance.username, password=make_password(instance.password))
            user.save()
            login(request, user)
            if form_medecin.is_valid():
                print(form_medecin)
                medecin = form_medecin.instance
                medecin.user = user
                medecin.save()
            return redirect("main:profile")
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
                return redirect("main:profil_medecin")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request=request,
                  template_name="main/login_medecin.html",
                  context={"login_form": form})


def profile(request):
    return render(request=request,
                  template_name="main/profile.html")


def logout_request(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("main:homepage")
