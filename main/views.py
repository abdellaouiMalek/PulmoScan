from django.shortcuts import render, redirect
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.utils import timezone
from django.conf import settings
import tempfile
from .ct_scan_utils import load_itk_image, convert_world_to_voxel_coord_bbox, resize_image_with_annotation_bbox

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

@login_required
def detection(request):
    """View for the nodules detection"""
    print("Detection view called")
    print(f"User: {request.user}")
    
    context = {
        'now': timezone.now(),
    }
    
    if request.method == 'POST':
        print("==== File Upload Request Received in Detection View ====")
        
        # Check if an .mhd file was uploaded
        if 'mhd_file' in request.FILES:
            mhd_file = request.FILES['mhd_file']
            
            # Log file information
            file_info = f"MHD File: {mhd_file.name}, Size: {mhd_file.size} bytes"
            print(file_info)
            
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            print(f"Created temporary directory: {temp_dir}")
            
            # Save file to get a path
            mhd_path = os.path.join(temp_dir, mhd_file.name)
            with open(mhd_path, 'wb') as f:
                for chunk in mhd_file.chunks():
                    f.write(chunk)
            
            # Save raw file if provided
            raw_path = None
            if 'raw_file' in request.FILES:
                raw_file = request.FILES['raw_file']
                raw_path = os.path.join(temp_dir, raw_file.name)
                with open(raw_path, 'wb') as f:
                    for chunk in raw_file.chunks():
                        f.write(chunk)
            
            # Load the CT scan using load_itk_image
            try:
                print(f"Loading ITK image from: {mhd_path}")
                numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_path)
                print(f"Image loaded successfully. Shape: {numpyImage.shape}")
                
                # Load annotations
                annotations = pd.read_csv('dep.csv')
                print(annotations.columns)
                
                # Extract seriesuid from filename
                filename = mhd_file.name
                seriesuid = os.path.splitext(filename)[0]
                print(f"Series UID: {seriesuid}")
                
                # Filter annotations for this series
                filtered_annotations = annotations[annotations["seriesuid"] == seriesuid]
                print(f"Filtered annotations: {len(filtered_annotations)} rows")
                
                # Check if there are any nodules in this scan
                has_nodules = not filtered_annotations.empty
                
                # Process nodule information if nodules exist
                nodules = []
                
                if has_nodules:
                    # First, update context with basic nodule information
                    for idx, row in filtered_annotations.iterrows():
                        # Extract nodule information
                        nodule = {
                            'id': idx + 1,
                            'x': row.get('coordX', 0),
                            'y': row.get('coordY', 0),
                            'z': row.get('coordZ', 0),
                            'diameter': row.get('diameter_mm', 0)
                        }
                        
                        # Convert world coordinates to voxel coordinates if needed
                        if all(k in row for k in ['coordX', 'coordY', 'coordZ']):
                            world_coords = np.array([row['coordZ'], row['coordY'], row['coordX']])
                            voxel_coords = np.round((world_coords - numpyOrigin) / numpySpacing).astype(int)
                            nodule['voxel_x'] = int(voxel_coords[2])
                            nodule['voxel_y'] = int(voxel_coords[1])
                            nodule['voxel_z'] = int(voxel_coords[0])
                        
                        nodules.append(nodule)
                    
                    # Update context with basic information
                    context.update({
                        'upload_success': True,
                        'mhd_file': mhd_file.name,
                        'mhd_path': mhd_path,
                        'raw_file': request.FILES['raw_file'].name if 'raw_file' in request.FILES else None,
                        'raw_path': raw_path if 'raw_file' in request.FILES else None,
                        'image_shape': numpyImage.shape,
                        'has_nodules': has_nodules,
                        'nodule_count': len(filtered_annotations),
                        'nodules': nodules,
                        'processing_status': 'initial',
                        'nodule_message': f"{len(filtered_annotations)} nodule(s) were found in this CT scan. Please consult with a medical professional for proper diagnosis."
                    })
                    
                    # Return the initial response with nodule information
                    return render(request, 'detection/detection.html', context)
                else:
                    # No nodules found, update context
                    context.update({
                        'upload_success': True,
                        'mhd_file': mhd_file.name,
                        'mhd_path': mhd_path,
                        'raw_file': request.FILES['raw_file'].name if 'raw_file' in request.FILES else None,
                        'raw_path': raw_path if 'raw_file' in request.FILES else None,
                        'image_shape': numpyImage.shape,
                        'has_nodules': False,
                        'nodule_count': 0,
                        'nodules': [],
                        'nodule_message': "No nodules were found in this CT scan. This indicates the scan is likely normal, but please consult with a medical professional for proper diagnosis."
                    })
                    
                    # Return the response for no nodules
                    return render(request, 'detection/detection.html', context)
                
            except Exception as e:
                print(f"Error loading or processing image: {str(e)}")
                import traceback
                traceback.print_exc()
                
                context.update({
                    'upload_success': False,
                    'error_message': f"Error processing image: {str(e)}",
                    'mhd_file': mhd_file.name,
                    'mhd_path': mhd_path,
                    'raw_file': request.FILES['raw_file'].name if 'raw_file' in request.FILES else None,
                    'raw_path': raw_path if 'raw_file' in request.FILES else None,
                })
    
    return render(request, 'detection/detection.html', context)

@login_required
def process_nodules(request):
    """Process nodules in a CT scan after initial detection"""
    context = {
        'now': timezone.now(),
    }
    
    if request.method == 'POST':
        # Get the paths from the form
        mhd_path = request.POST.get('mhd_path')
        seriesuid = request.POST.get('seriesuid')
        
        if not mhd_path or not os.path.exists(mhd_path):
            context.update({
                'error_message': "MHD file not found or invalid path.",
                'processing_status': 'error'
            })
            return render(request, 'detection/detection.html', context)
        
        try:
            # Load the CT scan
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(mhd_path)
            
            # Load annotations
            annotations = pd.read_csv('dep.csv')
            
            # Filter annotations for this series
            filtered_annotations = annotations[annotations["seriesuid"] == seriesuid]
            print(f"Series UID: {seriesuid}")
            print(f"Filtered annotations: {len(filtered_annotations)} rows")
            
            # Check if there are any nodules in this scan
            has_nodules = not filtered_annotations.empty
            
            if not has_nodules:
                context.update({
                    'error_message': "No nodules found for this scan.",
                    'processing_status': 'error'
                })
                return render(request, 'detection/detection.html', context)
            
            # Apply preprocessing to enhance nodule visibility
            from .ct_scan_utils import clip_CT_scan, isolate_lung, segment_nodules, visualize_single_slice
            from django.conf import settings
            import uuid
            
            # Clip the CT scan to standard lung window
            print("Applying CT scan clipping...")
            clipped_image = clip_CT_scan(numpyImage)
            
            # Apply lung isolation to focus on lung regions
            print("Applying lung isolation...")
            lung_mask = isolate_lung(clipped_image)
            
            # Apply nodule segmentation to identify potential nodule regions
            print("Applying nodule segmentation...")
            nodule_mask = segment_nodules(clipped_image)
            
            # Create directory for nodule slices
            nodule_dir = os.path.join(settings.MEDIA_ROOT, 'nodule_slices', seriesuid)
            os.makedirs(nodule_dir, exist_ok=True)
            
            # Prepare nodule information and visualize slices
            nodules = []
            nodule_slices = []
            
            print("Creating nodule slice visualizations...")
            for idx, row in filtered_annotations.iterrows():
                # Extract nodule information
                nodule = {
                    'id': idx + 1,
                    'x': row.get('coordX', 0),
                    'y': row.get('coordY', 0),
                    'z': row.get('coordZ', 0),
                    'diameter': row.get('diameter_mm', 0)
                }
                
                # Convert world coordinates to voxel coordinates
                world_coords = np.array([row['coordZ'], row['coordY'], row['coordX']])
                voxel_coords = np.round((world_coords - numpyOrigin) / numpySpacing).astype(int)
                nodule['voxel_x'] = int(voxel_coords[2])
                nodule['voxel_y'] = int(voxel_coords[1])
                nodule['voxel_z'] = int(voxel_coords[0])
                
                # Get the slice index (z-coordinate in voxel space)
                z_idx = voxel_coords[0]
                
                # Ensure z_idx is within bounds
                if z_idx < 0 or z_idx >= numpyImage.shape[0]:
                    continue
                
                # Create a unique filename
                unique_id = str(uuid.uuid4())[:8]
                filename = f'nodule_{idx+1}_slice_{z_idx}_{unique_id}.png'
                filepath = os.path.join(nodule_dir, filename)
                
                # Visualize the slice with the nodule highlighted
                visualize_single_slice(
                    clipped_image, 
                    filepath, 
                    z_idx, 
                    x=voxel_coords[2], 
                    y=voxel_coords[1]
                )
                
                # Add slice information to the list
                nodule_slices.append({
                    'id': idx + 1,
                    'filename': filename,
                    'url': f'/media/nodule_slices/{seriesuid}/{filename}',
                    'z_index': int(z_idx),
                    'diameter_mm': float(row['diameter_mm']),
                    'world_x': float(row['coordX']),
                    'world_y': float(row['coordY']),
                    'world_z': float(row['coordZ']),
                    'voxel_x': int(voxel_coords[2]),
                    'voxel_y': int(voxel_coords[1]),
                    'voxel_z': int(voxel_coords[0])
                })
                
                nodules.append(nodule)
            
            # Update context with processing status and slice information
            context.update({
                'upload_success': True,
                'mhd_file': os.path.basename(mhd_path),
                'mhd_path': mhd_path,
                'image_shape': numpyImage.shape,
                'has_nodules': has_nodules,
                'nodule_count': len(filtered_annotations),
                'processing_status': 'completed',
                'nodule_message': f"Preprocessing completed. {len(filtered_annotations)} nodule(s) were found and segmented.",
                'nodules': nodules,
                'nodule_slices': nodule_slices,
                'total_slices': numpyImage.shape[0]
            })
            
        except Exception as e:
            print(f"Error processing nodules: {str(e)}")
            import traceback
            traceback.print_exc()
            
            context.update({
                'error_message': f"Error processing nodules: {str(e)}",
                'processing_status': 'error'
            })
    
    return render(request, 'detection/detection.html', context)

def view_visualization(request, filename):
    """
    View for serving visualization files directly
    """
    from django.http import FileResponse, Http404
    import os
    from django.conf import settings
    
    # Construct the full path to the visualization file
    file_path = os.path.join(settings.MEDIA_ROOT, 'visualizations', filename)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Visualization file not found: {file_path}")
        raise Http404(f"Visualization file not found: {filename}")
    
    # Return the file as a response
    return FileResponse(open(file_path, 'rb'), content_type='text/html')
