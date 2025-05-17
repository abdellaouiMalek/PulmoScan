from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import UserProfile

class CustomUserCreationForm(UserCreationForm):
    """Custom user registration form with additional fields"""
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'}))
    first_name = forms.CharField(required=True, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Prénom'}))
    last_name = forms.CharField(required=True, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Nom'}))
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Bootstrap classes to form fields
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': "Nom d'utilisateur"})
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Mot de passe'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Confirmer le mot de passe'})
        
        # Customize help texts
        self.fields['username'].help_text = "Requis. 150 caractères maximum. Lettres, chiffres et @/./+/-/_ uniquement."
        self.fields['password1'].help_text = "Votre mot de passe doit contenir au moins 8 caractères et ne pas être trop commun."
        self.fields['password2'].help_text = "Entrez le même mot de passe que précédemment, pour vérification."

class CustomAuthenticationForm(AuthenticationForm):
    """Custom login form with Bootstrap styling"""
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': "Nom d'utilisateur"}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Mot de passe'}))

class UserProfileForm(forms.ModelForm):
    """Form for user profile information"""
    class Meta:
        model = UserProfile
        fields = ('title', 'institution', 'specialty', 'profile_image')
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Titre (ex: Dr., Prof.)'}),
            'institution': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Institution ou hôpital'}),
            'specialty': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Spécialité médicale'}),
            'profile_image': forms.FileInput(attrs={'class': 'form-control'}),
        }
