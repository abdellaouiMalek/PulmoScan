import os
import logging
import numpy as np
import pydicom
import SimpleITK as sitk
from django.conf import settings
import random
import pandas as pd
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def convert_world_to_voxel_coord_bbox(seriesuid, numpyOrigin, numpySpacing, annotations):
    """
    Convert world coordinates (in mm) to voxel coordinates.

    Parameters:
        seriesuid (str): Series UID for accessing annotations, can be None now.
        numpyOrigin (np.ndarray): The origin (z, y, x) of the scan in world space.
        numpySpacing (np.ndarray): Voxel spacing (z, y, x) in world space.
        annotations (pd.DataFrame or pd.Series): The annotations to process.

    Returns:
        pd.DataFrame: Updated annotations with voxel coordinates.
    """
    # Create a copy to avoid SettingWithCopyWarning
    annotations = annotations.copy()
    
    if seriesuid is not None:
        # Use .loc to ensure we're working with a proper DataFrame
        annotations = annotations.loc[annotations['seriesuid'] == seriesuid].copy()

    # Ensure DataFrame format
    if isinstance(annotations, pd.Series):
        annotations = annotations.to_frame().T

    # Convert coordinates
    annotations['coordZ'] = (annotations['coordZ'] - numpyOrigin[0]) / numpySpacing[0]
    annotations['coordY'] = (annotations['coordY'] - numpyOrigin[1]) / numpySpacing[1]
    annotations['coordX'] = (annotations['coordX'] - numpyOrigin[2]) / numpySpacing[2]

    # Convert bounding box coordinates
    annotations['bbox_z'] = (annotations['bbox_z'] - numpyOrigin[0]) / numpySpacing[0]
    annotations['bbox_y'] = (annotations['bbox_y'] - numpyOrigin[1]) / numpySpacing[1]
    annotations['bbox_x'] = (annotations['bbox_x'] - numpyOrigin[2]) / numpySpacing[2]

    # Convert diameter to voxel space
    annotations['diameter_voxel_z'] = annotations['diameter_mm'] / numpySpacing[0]
    annotations['diameter_voxel_y'] = annotations['diameter_mm'] / numpySpacing[1]
    annotations['diameter_voxel_x'] = annotations['diameter_mm'] / numpySpacing[2]

    return annotations

def resize_image_with_annotation_bbox(seriesuid, numpyImage, numpyOrigin, numpySpacing, annotations):
    """
    Resize a 3D CT scan to 1mm voxel spacing and update corresponding annotations.

    Parameters:
        seriesuid (str): The seriesuid of the scan.
        numpyImage (np.ndarray): 3D volume (z, y, x).
        numpyOrigin (np.ndarray): Original origin (z, y, x).
        numpySpacing (np.ndarray): Original voxel spacing (z, y, x).
        annotations (pd.DataFrame or pd.Series): Annotations containing nodule center and bounding box.

    Returns:
        tuple: (resized volume, new origin, new spacing, updated annotations)
    """
    # Step 1: Convert world coordinates to voxel coordinates first
    newAnnotations = convert_world_to_voxel_coord_bbox(seriesuid, numpyOrigin, numpySpacing, annotations).copy()

    # Step 2: Define target spacing
    RESIZE_SPACING = np.array([1, 1, 1])

    # Step 3: Compute resize factors
    resize_factor = numpySpacing / RESIZE_SPACING
    new_real_shape = numpyImage.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    real_resize = new_shape / numpyImage.shape

    # Step 4: Resize the image
    new_volume = ndimage.zoom(numpyImage, zoom=real_resize, order=1)  # linear interpolation (order=1)

    # Step 5: Adjust the origin properly (origin stays the same in world space)
    newOrigin = numpyOrigin  # No change needed to the origin after resampling spacing!

    # Step 6: Update the annotations (rescale voxel coordinates)
    scale = np.array([resize_factor[0], resize_factor[1], resize_factor[2]])

    # Create a copy of the columns we'll modify
    for coord in ["coordZ", "bbox_z"]:
        newAnnotations.loc[:, coord] = (newAnnotations[coord] * scale[0]).round().astype(int)

    for coord in ["coordY", "bbox_y"]:
        newAnnotations.loc[:, coord] = (newAnnotations[coord] * scale[1]).round().astype(int)

    for coord in ["coordX", "bbox_x"]:
        newAnnotations.loc[:, coord] = (newAnnotations[coord] * scale[2]).round().astype(int)

    return new_volume, newOrigin, RESIZE_SPACING, newAnnotations

def clip_CT_scan(numpyImage):
    return np.clip(numpyImage, -1200, 600)

def threshHold_segmentation(numpyImage, lower=-1200, upper=-600):
    """
    Segment the lung region using thresholding.
    Args:
        numpyImage: Input image in Hounsfield Units (HU).
        lower: Lower Hounsfield value
        upp: Upper Hounsfield value
    Returns:
        Binary mask of the slected region.
    """
    mask = np.logical_and(numpyImage >= lower, numpyImage <= upper)

    return mask

def get_lung_mask(numpyImage):
    # Get the lung region
    lung_mask = threshHold_segmentation(numpyImage)

    # Post process the lung mask (remove holes/smooth boundaries)
    lung_mask = morphology.binary_closing(lung_mask, morphology.ball(5))
    lung_mask = morphology.binary_opening(lung_mask, morphology.ball(5))
    lung_mask = morphology.binary_dilation(lung_mask, morphology.ball(2))

    return lung_mask

def refine_nodule_masks(
        numpyImage, 
        lung_mask, 
        nodule_ranges= [
            (-1200, -600),  # Lung tissue
            (-600, -300),  # Subsolid nodules (Ground-Glass Opacities)
            (-100, 100),   # Solid nodules
            (300, 600)     # Calcified nodules
        ]
    ):
    refined_masks = []

    for lower, upper in nodule_ranges:
        nodule_mask = threshHold_segmentation(numpyImage, lower=lower, upper=upper)
        # Restrict to lung region
        refined_mask = np.logical_and(nodule_mask, lung_mask)
        refined_masks.append(refined_mask.astype(np.uint8))
    
    return refined_masks

def isolate_lung(numpyImage):
    lung_mask = get_lung_mask(numpyImage)
    refined_masks = refine_nodule_masks(numpyImage, lung_mask)

    combined_mask = np.zeros_like(refined_masks[0])
    for mask in refined_masks:
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask

def Min_Max_scaling(numpyImage):
    normalized_image = (numpyImage - (-1200)) / (600 - (-1000))

    return normalized_image

def segment_nodules(ct_scan):
    # Combine masks for all nodule types
    mask = np.zeros_like(ct_scan, dtype=bool)

    ranges = [
        (-600, -300),  # subsolid
        (-100, 100),   # solid
        (300, 600)     # calcified
    ]

    for low, high in ranges:
        mask |= (ct_scan >= low) & (ct_scan <= high)

    return mask.astype(np.uint8)

def create_3d_visualization(ct_scan, lung_mask, nodule_mask, annotations, origin, spacing, seriesuid):
    """
    Create a 3D visualization of the CT scan with highlighted nodules and semi-transparent lung parts
    using Three.js for better rendering and interactivity.
    """
    import os
    import json
    import numpy as np
    from skimage import measure
    from django.conf import settings
    
    # Create directory for visualizations if it doesn't exist
    media_dir = os.path.join(settings.MEDIA_ROOT, 'visualizations')
    os.makedirs(media_dir, exist_ok=True)
    
    # Generate a unique filename
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    filename = f'visualization_{seriesuid}_{unique_id}.html'
    html_path = os.path.join(media_dir, filename)
    json_filename = f'data_{seriesuid}_{unique_id}.json'
    json_path = os.path.join(media_dir, json_filename)
    
    # Create the correct URL path for the visualization
    relative_path = f'/media/visualizations/{filename}'
    json_url = f'/media/visualizations/{json_filename}'
    
    # Downsample the data to make it more manageable for visualization
    step = 2
    downsampled_lung = lung_mask[::step, ::step, ::step]
    downsampled_nodule = nodule_mask[::step, ::step, ::step]
    
    # Extract lung surface using marching cubes algorithm
    print("Extracting lung surface...")
    lung_verts, lung_faces, _, _ = measure.marching_cubes(downsampled_lung, level=0.5)
    
    # Scale vertices back to original dimensions
    lung_verts = lung_verts * step * spacing
    
    # Extract nodule surface
    print("Extracting nodule surface...")
    try:
        nodule_verts, nodule_faces, _, _ = measure.marching_cubes(downsampled_nodule, level=0.5)
        # Scale vertices back to original dimensions
        nodule_verts = nodule_verts * step * spacing
        has_nodule_mesh = True
    except:
        print("Could not extract nodule surface - may be too small or not present in downsampled data")
        has_nodule_mesh = False
        nodule_verts = np.array([])
        nodule_faces = np.array([])
    
    # Prepare data for Three.js
    lung_geometry = {
        'vertices': lung_verts.flatten().tolist(),
        'faces': lung_faces.flatten().tolist()
    }
    
    nodule_geometry = {
        'vertices': nodule_verts.flatten().tolist() if has_nodule_mesh else [],
        'faces': nodule_faces.flatten().tolist() if has_nodule_mesh else []
    }
    
    # Prepare annotated nodules data
    annotated_nodules = []
    for idx, row in annotations.iterrows():
        if all(k in row for k in ['coordX', 'coordY', 'coordZ', 'diameter_mm']):
            annotated_nodules.append({
                'id': int(idx),
                'x': float(row['coordX']),
                'y': float(row['coordY']),
                'z': float(row['coordZ']),
                'diameter': float(row['diameter_mm'])
            })
    
    # Create JSON data for Three.js
    visualization_data = {
        'lung': lung_geometry,
        'nodule': nodule_geometry,
        'annotatedNodules': annotated_nodules
    }
    
    # Save JSON data
    with open(json_path, 'w') as f:
        json.dump(visualization_data, f)
    
    # Create HTML file with Three.js visualization
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CT Scan 3D Visualization</title>
        <style>
            body {{ margin: 0; overflow: hidden; }}
            canvas {{ width: 100%; height: 100%; display: block; }}
            #info {{
                position: absolute;
                top: 10px;
                width: 100%;
                text-align: center;
                color: white;
                font-family: Arial, sans-serif;
                pointer-events: none;
                text-shadow: 1px 1px 1px black;
            }}
            #legend {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                background-color: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                margin-right: 10px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div id="info">CT Scan 3D Visualization: {seriesuid}</div>
        <div id="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: lightblue;"></div>
                <div>Lung Tissue</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: red;"></div>
                <div>Segmented Nodules</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: yellow;"></div>
                <div>Annotated Nodules</div>
            </div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r146/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.146.0/examples/js/controls/OrbitControls.js"></script>
        <script>
            // Main Three.js code
            let scene, camera, renderer, controls;
            let lungMesh, noduleMesh, annotatedNodules = [];
            
            // Initialize the scene
            function init() {{
                // Create scene
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x111111);
                
                // Create camera
                camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.z = 200;
                
                // Create renderer
                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);
                
                // Add controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.25;
                
                // Add lights
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(0, 1, 1);
                scene.add(directionalLight);
                
                // Load data and create meshes
                loadData();
                
                // Handle window resize
                window.addEventListener('resize', onWindowResize);
            }}
            
            // Load data from JSON
            function loadData() {{
                fetch('{json_url}')
                    .then(response => response.json())
                    .then(data => {{
                        createLungMesh(data.lung);
                        createNoduleMesh(data.nodule);
                        createAnnotatedNodules(data.annotatedNodules);
                        
                        // Center camera on the lung
                        centerCamera();
                    }})
                    .catch(error => console.error('Error loading data:', error));
            }}
            
            // Create lung mesh
            function createLungMesh(lungData) {{
                if (lungData.vertices.length === 0) return;
                
                const geometry = new THREE.BufferGeometry();
                
                // Set vertices
                const vertices = new Float32Array(lungData.vertices);
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                
                // Set faces
                const indices = new Uint32Array(lungData.faces);
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                
                // Compute normals for proper lighting
                geometry.computeVertexNormals();
                
                // Create material with transparency
                const material = new THREE.MeshPhongMaterial({{
                    color: 0x87ceeb,  // lightblue
                    transparent: true,
                    opacity: 0.3,
                    side: THREE.DoubleSide
                }});
                
                // Create mesh and add to scene
                lungMesh = new THREE.Mesh(geometry, material);
                scene.add(lungMesh);
            }}
            
            // Create nodule mesh
            function createNoduleMesh(noduleData) {{
                if (noduleData.vertices.length === 0) return;
                
                const geometry = new THREE.BufferGeometry();
                
                // Set vertices
                const vertices = new Float32Array(noduleData.vertices);
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                
                // Set faces
                const indices = new Uint32Array(noduleData.faces);
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                
                // Compute normals for proper lighting
                geometry.computeVertexNormals();
                
                // Create material
                const material = new THREE.MeshPhongMaterial({{
                    color: 0xff0000,  // red
                    transparent: true,
                    opacity: 0.8,
                    side: THREE.DoubleSide
                }});
                
                // Create mesh and add to scene
                noduleMesh = new THREE.Mesh(geometry, material);
                scene.add(noduleMesh);
            }}
            
            // Create annotated nodules as spheres
            function createAnnotatedNodules(nodules) {{
                nodules.forEach(nodule => {{
                    const geometry = new THREE.SphereGeometry(nodule.diameter / 2, 32, 32);
                    const material = new THREE.MeshPhongMaterial({{
                        color: 0xffff00,  // yellow
                        transparent: true,
                        opacity: 0.9
                    }});
                    
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(nodule.x, nodule.y, nodule.z);
                    
                    scene.add(sphere);
                    annotatedNodules.push(sphere);
                }});
            }}
            
            // Center camera on the lung
            function centerCamera() {{
                if (lungMesh) {{
                    // Compute bounding box
                    lungMesh.geometry.computeBoundingBox();
                    const box = lungMesh.geometry.boundingBox;
                    
                    // Create a Box3Helper to visualize the bounding box
                    const helper = new THREE.Box3Helper(box, 0xffff00);
                    scene.add(helper);
                    
                    // Get center and size of the bounding box
                    const center = new THREE.Vector3();
                    box.getCenter(center);
                    const size = new THREE.Vector3();
                    box.getSize(size);
                    
                    // Position camera to see the entire lung
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const fov = camera.fov * (Math.PI / 180);
                    const cameraDistance = maxDim / (2 * Math.tan(fov / 2));
                    
                    camera.position.copy(center);
                    camera.position.z += cameraDistance * 1.5;
                    camera.lookAt(center);
                    
                    // Update controls target
                    controls.target.copy(center);
                    controls.update();
                }}
            }}
            
            // Handle window resize
            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            
            // Start the visualization
            init();
            animate();
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"3D visualization saved to {html_path}")
    print(f"Relative path for browser: {relative_path}")
    
    return relative_path

def highlight_nodules_in_slices(ct_scan, annotations, origin, spacing, seriesuid, output_dir=None):
    """
    Extract and highlight nodules in CT scan slices.
    
    Args:
        ct_scan: 3D numpy array of the CT scan
        annotations: DataFrame with nodule annotations
        origin: Origin coordinates of the CT scan
        spacing: Spacing of the CT scan
        seriesuid: Series UID for the CT scan
        output_dir: Directory to save the images (default: media/nodule_slices)
        
    Returns:
        List of dictionaries with slice information
    """
    import os
    import numpy as np
    # Set the backend to 'Agg' before importing pyplot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from django.conf import settings
    import uuid
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(settings.MEDIA_ROOT, 'nodule_slices', seriesuid)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Window the CT scan for better visualization
    ct_windowed = clip_CT_scan(ct_scan)
    
    # List to store slice information
    slice_info = []
    
    # Process each nodule
    for idx, row in annotations.iterrows():
        # Get nodule world coordinates
        world_coords = np.array([row['coordZ'], row['coordY'], row['coordX']])
        
        # Convert to voxel coordinates
        voxel_coords = np.round((world_coords - origin) / spacing).astype(int)
        
        # Get nodule diameter in voxels
        diameter_mm = row['diameter_mm']
        diameter_voxels = np.round(diameter_mm / spacing).astype(int)
        
        # Get the slice index (z-coordinate in voxel space)
        z_idx = voxel_coords[0]
        
        # Ensure z_idx is within bounds
        if z_idx < 0 or z_idx >= ct_scan.shape[0]:
            continue
        
        # Extract the slice
        ct_slice = ct_windowed[z_idx, :, :]
        
        # Create a unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f'nodule_{idx+1}_slice_{z_idx}_{unique_id}.png'
        filepath = os.path.join(output_dir, filename)
        
        # Create figure with three views: axial, coronal, and sagittal
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Axial view (xy plane)
        axes[0].imshow(ct_slice, cmap='gray')
        circle = Circle((voxel_coords[2], voxel_coords[1]), 
                       diameter_voxels[2]/2, 
                       fill=False, 
                       edgecolor='red', 
                       linewidth=2)
        axes[0].add_patch(circle)
        axes[0].set_title(f'Axial (Z={z_idx})')
        axes[0].axis('off')
        
        # Coronal view (xz plane)
        y_idx = voxel_coords[1]
        coronal_slice = ct_windowed[:, y_idx, :]
        axes[1].imshow(coronal_slice, cmap='gray')
        circle = Circle((voxel_coords[2], voxel_coords[0]), 
                       diameter_voxels[2]/2, 
                       fill=False, 
                       edgecolor='red', 
                       linewidth=2)
        axes[1].add_patch(circle)
        axes[1].set_title(f'Coronal (Y={y_idx})')
        axes[1].axis('off')
        
        # Sagittal view (yz plane)
        x_idx = voxel_coords[2]
        sagittal_slice = ct_windowed[:, :, x_idx]
        axes[2].imshow(sagittal_slice, cmap='gray')
        circle = Circle((voxel_coords[1], voxel_coords[0]), 
                       diameter_voxels[1]/2, 
                       fill=False, 
                       edgecolor='red', 
                       linewidth=2)
        axes[2].add_patch(circle)
        axes[2].set_title(f'Sagittal (X={x_idx})')
        axes[2].axis('off')
        
        # Add overall title
        plt.suptitle(f'Nodule {idx+1}: {diameter_mm:.1f}mm diameter at ({row["coordX"]:.1f}, {row["coordY"]:.1f}, {row["coordZ"]:.1f}) mm', 
                    fontsize=14)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Add slice information to the list
        slice_info.append({
            'id': idx + 1,
            'filename': filename,
            'url': f'/media/nodule_slices/{seriesuid}/{filename}',
            'z_index': int(z_idx),
            'diameter_mm': float(diameter_mm),
            'world_x': float(row['coordX']),
            'world_y': float(row['coordY']),
            'world_z': float(row['coordZ']),
            'voxel_x': int(voxel_coords[2]),
            'voxel_y': int(voxel_coords[1]),
            'voxel_z': int(voxel_coords[0])
        })
    
    return slice_info

def extract_ct_slices(ct_scan, seriesuid, num_slices=10, output_dir=None):
    """
    Extract and save representative slices from the CT scan.
    
    Args:
        ct_scan: 3D numpy array of the CT scan
        seriesuid: Series UID for the CT scan
        num_slices: Number of slices to extract (default: 10)
        output_dir: Directory to save the images (default: media/ct_slices)
        
    Returns:
        List of dictionaries with slice information
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from django.conf import settings
    import uuid
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(settings.MEDIA_ROOT, 'ct_slices', seriesuid)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Window the CT scan for better visualization
    ct_windowed = clip_CT_scan(ct_scan)
    
    # Calculate slice indices to extract (evenly distributed)
    total_slices = ct_scan.shape[0]
    if num_slices >= total_slices:
        # If requested slices exceed total, take all slices
        slice_indices = range(total_slices)
    else:
        # Otherwise, take evenly distributed slices
        step = total_slices // num_slices
        slice_indices = range(step // 2, total_slices, step)[:num_slices]
    
    # List to store slice information
    slice_info = []
    
    # Extract and save each slice
    for i, z_idx in enumerate(slice_indices):
        # Extract the slice
        ct_slice = ct_windowed[z_idx, :, :]
        
        # Create a unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f'slice_{z_idx}_{unique_id}.png'
        filepath = os.path.join(output_dir, filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_slice, cmap='gray')
        ax.set_title(f'Slice {z_idx}')
        ax.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Add slice information to the list
        slice_info.append({
            'index': int(z_idx),
            'filename': filename,
            'url': f'/media/ct_slices/{seriesuid}/{filename}'
        })
    
    return slice_info



def visualize_single_slice(volumeImage, plot_name, slice, x=None, y=None):
    """
    Visualizes a single slice from a volume image and saves it to a file.
    
    Parameters:
        volumeImage (ndarray): 3D volume data.
        plot_name (str): Path to save the plot.
        slice (int): Index of the slice to visualize.
        x (float or int, optional): X-coordinate for the red dot.
        y (float or int, optional): Y-coordinate for the red dot.
    
    Returns:
        str: Path to the saved image
    """
    # Set the backend to 'Agg' before creating the figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    slice = int(round(slice))

    # Display the slice (CT scan) in grayscale
    im = ax.imshow(volumeImage[slice], cmap="gray")
    ax.set_title(f"CT Scan Slice {slice}")
    ax.axis("off")
    
    # If both x and y coordinates are provided, plot a red dot and circle
    if x is not None and y is not None:
        # Add a red dot at the nodule location
        ax.plot(x, y, 'ro', markersize=8)
        
        # Add a circle to represent the nodule (with a fixed size for now)
        from matplotlib.patches import Circle
        circle = Circle((x, y), 15, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(circle)
        
        # Add a label
        ax.annotate("Nodule", (x, y), color="white", fontsize=10, 
                   xytext=(5, 5), textcoords="offset points",
                   bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7))

    # Ensure the directory exists
    os.makedirs(os.path.dirname(plot_name), exist_ok=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_name




