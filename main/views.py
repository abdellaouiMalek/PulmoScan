from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.utils import timezone

def home(request):
    """View for the home page"""
    return render(request, 'home.html')

def about(request):
    """View for the about page"""
    return render(request, 'about.html')

def services(request):
    """View for the services page"""
    return render(request, 'services.html')

def doctors(request):
    """View for the doctors page"""
    return render(request, 'doctors.html')

def contact(request):
    """View for the contact page"""
    return render(request, 'contact.html')

def appointment(request):
    """View for the appointment page"""
    return render(request, 'appointment.html')

def test(request):
    """View for the test page"""
    return render(request, 'test.html')

@login_required
def dashboard(request):
    """View for the user dashboard"""
    print("Dashboard view called")
    print(f"User: {request.user}")
    context = {
        'now': timezone.now(),
    }
    return render(request, 'dashboard.html', context)

@login_required
def profile(request):
    """View for the user profile"""
    return render(request, 'profile.html')

def login_view(request):
    """View for the login page"""
    print("Login view called")
    print(f"Request method: {request.method}")

    # If user is already logged in, redirect to dashboard
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        print("Processing POST request")
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(f"Username: {username}")
        print(f"Password: {'*' * len(password)}")

        # Try to authenticate with username
        user = authenticate(request, username=username, password=password)

        # If authentication fails, try with email
        if user is None:
            try:
                user_obj = User.objects.get(email=username)
                user = authenticate(request, username=user_obj.username, password=password)
            except User.DoesNotExist:
                user = None

        print(f"Authentication result: {user}")

        if user is not None:
            print("User authenticated successfully")
            login(request, user)
            messages.success(request, f"Welcome back, {user.first_name if user.first_name else user.username}!")
            print("Redirecting to dashboard")

            # Redirect to dashboard
            return redirect('dashboard')
        else:
            print("Authentication failed")
            messages.error(request, "Invalid username/email or password.")

    return render(request, 'login.html')

def register(request):
    """View for the registration page"""
    print("Register view called")
    print(f"Request method: {request.method}")

    if request.method == 'POST':
        print("Processing POST request")
        username = request.POST.get('username')
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"First name: {first_name}")
        print(f"Last name: {last_name}")

        # Check if passwords match
        if password1 != password2:
            print("Passwords don't match")
            messages.error(request, "Passwords don't match.")
            return render(request, 'register.html')

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            print("Username already exists")
            messages.error(request, "Username already exists.")
            return render(request, 'register.html')

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            print("Email already exists")
            messages.error(request, "Email already exists.")
            return render(request, 'register.html')

        # Create user
        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password1,
                first_name=first_name,
                last_name=last_name
            )
            print(f"User created: {user}")
            messages.success(request, "Account created successfully. You can now login.")
            return redirect('login')
        except Exception as e:
            print(f"Error creating user: {e}")
            messages.error(request, f"Error creating account: {e}")

    return render(request, 'register.html')

def logout_view(request):
    """View for logging out"""
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('home')
