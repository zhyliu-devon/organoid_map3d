import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
from PIL import Image
from scipy.interpolate import Rbf
import math
# Function to convert spherical coordinates (lat, lon) to 2D coordinates (equirectangular projection)
def spherical_to_2d(lat, lon):
    x = lon
    y = lat
    return x, y

# Function to create pole data with a specified longitude range
def create_pole_data(lat, num_points, latency_value, longitude_range):
    longitudes = np.linspace(longitude_range[0], longitude_range[1], num_points)
    latitudes = np.full(num_points, lat)
    latency = np.full(num_points, latency_value)
    return latitudes, longitudes, latency

def lon_lat_to_azimuthal_equidistant(longitudes, latitudes, center_lon, center_lat):
    # Convert degrees to radians
    lambda_ = np.radians(longitudes)
    phi = np.radians(latitudes)
    lambda_0 = np.radians(center_lon)
    phi_0 = np.radians(center_lat)
    
    # Calculate the difference in longitude
    delta_lambda = lambda_ - lambda_0
    
    # Calculate the angular distance using the spherical law of cosines
    theta = np.arccos(np.sin(phi_0) * np.sin(phi) + np.cos(phi_0) * np.cos(phi) * np.cos(delta_lambda))
    
    # Calculate the 2D coordinates
    x = theta * np.sin(delta_lambda)
    y = -theta * np.cos(delta_lambda)
    
    # Convert the radius to a more familiar unit if needed (e.g., kilometers)
    # Assuming Earth's radius is approximately 6371 kilometers
    radius = 100
    x *= radius
    y *= radius
    return x, y

def find_value_in_grid(x,y,z):
    points = np.array([x.flatten(), y.flatten()]).T
    values = z.flatten()

    # Define the point where you want to interpolate
    point_of_interest = np.array([[0, 0]])

    # Perform the interpolation
    z_value_at_point = griddata(points, values, point_of_interest, method='linear')
    return z_value_at_point

def calculate_pole_data_by_interpolation(latency,target_indeces):
    latitudes = np.array([ 0,   60,-60,   0,     0, 60, -60,     0, 0    ,-60, 60, 0   ,     0,-60,  60,      0])
    longitudes = np.array([22.5, 0, 90,67.5, 112.5, 90, 180, 157.5, -22.5, 0 ,-90,-67.5,-112.5,-90, 180, -157.5])
    latitudes = latitudes[target_indeces]
    longitudes = longitudes[target_indeces]
    x_2d_north, y_2d_north = lon_lat_to_azimuthal_equidistant(
        longitudes,latitudes, 0,90
    )
    x_2d_south, y_2d_south = lon_lat_to_azimuthal_equidistant(
        longitudes,latitudes, 0,-90
    )
    coords_north = {"x2d_full": x_2d_north,
        "y2d_full": y_2d_north,"full_latency": latency,
        "latitudes": latitudes,"longitudes": longitudes,}
    coords_south = {"x2d_full": x_2d_south,
        "y2d_full": y_2d_south,"full_latency": latency,
        "latitudes": latitudes,"longitudes": longitudes,}
    grid_north = grid_fitting(coords_north, save = 0, plot = 0)
    north_x = grid_north['grid_x_full']
    north_y = grid_north['grid_y_full']
    north_z = grid_north['grid_z_full']
    north = find_value_in_grid(north_x,north_y,north_z)

    grid_south = grid_fitting(coords_south, save = 0, plot = 0)
    south_x = grid_south['grid_x_full']
    south_y = grid_south['grid_y_full']
    south_z = grid_south['grid_z_full']
    south = find_value_in_grid(south_x,south_y,south_z)

    return (north*10,south*10)

# Function to adjust longitudes for extended data
def adjust_longitudes(longitudes, shift):
    adjusted = longitudes + shift
    return adjusted

def create_coordinates(latency, target_indeces, pole_latency = -1, adjusted_coord = None):
    #latency = np.array(target_data[target_indeces,i*int(fs_new/samplingRate)])
    latitudes = np.array([ 0,   60,-60,   0,     0, 60, -60,     0, 0    ,-60, 60, 0   ,     0,-60,  60,      0])
    longitudes = np.array([22.5, 0, 90,67.5, 112.5, 90, 180, 157.5, -22.5, 0 ,-90,-67.5,-112.5,-90, 180, -157.5])
    if adjusted_coord is not None:
        latitudes = adjusted_coord['lat']
        longitudes = adjusted_coord['lon']
    latitudes = latitudes[target_indeces]
    longitudes = longitudes[target_indeces]

    
    # Extend the data by appending three sets to both sides
    extended_latitudes = np.concatenate([latitudes]*5)
    extended_longitudes = np.concatenate([adjust_longitudes(longitudes, i * -360) for i in range(-2, 3)])
    extended_latency = np.concatenate([latency]*5)

    latitudes_mirrored_top = -extended_latitudes + 180  
    latitudes_mirrored_down = -extended_latitudes - 180
    longitudes_mirrored_top = extended_longitudes - 180
    longitudes_mirrored_down = extended_longitudes - 180

    # Define the range of longitudes for the extended pole data and create pole data
    num_pole_points = 100
    extended_longitude_range = [np.min(extended_longitudes), np.max(extended_longitudes)]
    if (pole_latency == -1):
        pole_latitudes_up, pole_longitudes_up, pole_latency_up = create_pole_data(90, num_pole_points, latency[latitudes.argmax()], extended_longitude_range)
        pole_latitudes_down, pole_longitudes_down, pole_latency_down = create_pole_data(-90, num_pole_points, latency[latitudes.argmin()], extended_longitude_range)
    else:
        pole_latitudes_up, pole_longitudes_up, pole_latency_up = create_pole_data(90, num_pole_points, pole_latency[0], extended_longitude_range)
        pole_latitudes_down, pole_longitudes_down, pole_latency_down = create_pole_data(-90, num_pole_points, pole_latency[1], extended_longitude_range)        
    # Combine extended data with pole data
    full_latitudes = np.concatenate([extended_latitudes, latitudes_mirrored_top, latitudes_mirrored_down, pole_latitudes_up, pole_latitudes_down])
    full_longitudes = np.concatenate([extended_longitudes, longitudes_mirrored_top, longitudes_mirrored_down, pole_longitudes_up, pole_longitudes_down])
    full_latency = np.concatenate([extended_latency, extended_latency,extended_latency,pole_latency_up, pole_latency_down])

    # Convert to 2D
    x2d_full, y2d_full = spherical_to_2d(full_latitudes, full_longitudes)

    x2d, y2d = spherical_to_2d(latitudes,longitudes)
    coords = {
        "x2d_full": x2d_full,
        "y2d_full": y2d_full,
        "full_latency": full_latency,
        "full_latitudes": full_latitudes,
        "full_longitudes": full_longitudes,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "x2d": x2d,
        "y2d": y2d,
    }
    return coords
    
def grid_fitting(*args, **kwargs):
    if args and isinstance(args[0], dict):
        # If the first argument is a dictionary, extract all values from it
        coords = args[0]
        x2d_full = coords.get('x2d_full', None)
        y2d_full = coords.get('y2d_full', None)
        full_latency = coords.get('full_latency', None)
        # full_latitudes = coords.get('full_latitudes', None)
        # full_longitudes = coords.get('full_longitudes', None)
        latitudes = coords.get('latitudes', None)
        longitudes = coords.get('longitudes', None)
    else:
        # Otherwise, extract individual keyword arguments, defaulting to None if not provided
        x2d_full = kwargs.get('x2d_full', None)
        y2d_full = kwargs.get('y2d_full', None)
        full_latency = kwargs.get('full_latency', None)
        # full_latitudes = kwargs.get('full_latitudes', None)
        # full_longitudes = kwargs.get('full_longitudes', None)
        latitudes = kwargs.get('latitudes', None)
        longitudes = kwargs.get('longitudes', None)
    full_latency = full_latency/10
    save = kwargs.pop('save', 1)
    file_address = kwargs.pop('file_name', 'contour.png')
    plot = kwargs.pop('plot', 0)
    colorbar = kwargs.pop('colorbar', 0)
    electrodes = kwargs.pop('electrodes', 1)
    dense = kwargs.pop('dense', 1)
    color = kwargs.pop('color', 'black')

    # Create grid for interpolation
    grid_x_full, grid_y_full = np.mgrid[-540:540:dense*1800j, -270:270:dense*900j]

    coords = np.column_stack((x2d_full, y2d_full))
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    x2d_unique = x2d_full#[unique_indices]
    y2d_unique = y2d_full#[unique_indices]
    full_latency_unique = full_latency#[unique_indices]
    # Interpolate
    rbf = Rbf(x2d_unique,y2d_unique, full_latency_unique, function='multiquadric')
    grid_z_full = rbf(grid_x_full, grid_y_full)
    # Plotting
    x2d, y2d = spherical_to_2d(latitudes,longitudes)

    # Example array
    array = np.array(grid_x_full[:,0])
    array_y = np.array(grid_y_full[0,:])
    # Find indices where the condition is true
    indices = np.where((array >= -160.3) & (array <= 200.2))[0]
    if 1:
        indices_Y = np.where((array_y >= -90.2) & (array_y <= 90.2))[0]

        grid_x_target = grid_x_full[indices][:,indices_Y ]
        grid_y_target = grid_y_full[indices][:,indices_Y]
        grid_z_target = grid_z_full[indices][:,indices_Y]
        print(grid_x_target.shape)
    else:
        grid_x_target = grid_x_full[indices]
        grid_y_target = grid_y_full[indices]
        grid_z_target = grid_z_full[indices]
        print(grid_x_target.shape)
    # grid_x_target = grid_x_full
    # grid_y_target = grid_y_full
    # grid_z_target = grid_z_full
    # Create a figure with a specific aspect ratio (2:1 width:height)
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed
    
    # Plot the data
    contour = ax.contourf(grid_x_target, grid_y_target, grid_z_target, levels=15, cmap='jet')
    if(electrodes):
        plt.scatter(x2d, y2d, c=color, edgecolor='k') 


    
    if save:
        ax.axis('off')  # Ensures no axis information is included in the output image
        plt.tight_layout(pad=0)  # Reduces or removes padding around the plot
        # Save the figure with adjusted settings to eliminate white space
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(file_address, bbox_inches='tight', pad_inches=0, dpi=600)
        #plt.show()
        plt.close(fig)  # Close the plot to free up memory

    elif(plot):
        plt.axis('equal')
        plt.xlim([-160,200])
        plt.ylim([-90,90])

        #plt.tight_layout(pad=0)
        if(colorbar):
            colorbar = plt.colorbar(contour)
            colorbar.set_label('Latency (ms)')
            plt.show()
    if not plot:
        plt.close()
    grid_variables = {
    "grid_x_full": grid_x_full,
    "grid_y_full": grid_y_full,
    "grid_z_full": grid_z_full,
    "grid_x_target": grid_x_target,
    "grid_y_target": grid_y_target,
    "grid_z_target": grid_z_target
    }

    return grid_variables

def create_window(filename):
        # Assume image_path is defined and points to an existing image file
    image_path = filename  # Example path, adjust as necessary
    img = Image.open(image_path).convert('RGB')
    img_data = np.flipud(np.array(img))

    # Convert image data to a VTK compatible format
    height, width, _ = img_data.shape
    img_data_2d = img_data.reshape(height*width, 3)
    vtk_array = numpy_to_vtk(num_array=img_data_2d, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    # Create a VTK image and texture
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, 1)
    vtk_image.GetPointData().SetScalars(vtk_array)
    texture = vtk.vtkTexture()
    texture.SetInputData(vtk_image)
    texture.InterpolateOn()

    # Create the geometry and actor to apply the texture
    sphere = vtk.vtkTexturedSphereSource()
    sphere.SetThetaResolution(500)
    sphere.SetPhiResolution(500)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere.GetOutputPort())
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.SetTexture(texture)

    # Setup renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1920, 1080)
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add actor to the scene and configure camera
    renderer.AddActor(sphere_actor)
    renderer.SetBackground(0, 0, 0)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(8, 0, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    # Start the interaction
    render_window.Render()
    render_window_interactor.Initialize()
    render_window_interactor.Start()


import os
import vtk
from PIL import Image
import numpy as np
def take_360_screenshot(filename):
    image_path = filename
    img = Image.open(image_path).convert('RGB')
    img_data = np.flipud(np.array(img))

    height, width, _ = img_data.shape
    img_data_2d = img_data.reshape(height * width, 3)
    vtk_array = numpy_to_vtk(num_array=img_data_2d, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, 1)
    vtk_image.GetPointData().SetScalars(vtk_array)
    texture = vtk.vtkTexture()
    texture.SetInputData(vtk_image)
    texture.InterpolateOn()

    sphere = vtk.vtkTexturedSphereSource()
    sphere.SetThetaResolution(500)
    sphere.SetPhiResolution(500)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere.GetOutputPort())
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.SetTexture(texture)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1920, 1080)
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(sphere_actor)
    renderer.SetBackground(0, 0, 0)
    camera = renderer.GetActiveCamera()

    # Ensure the directory exists
    directory = '3dMappingPic'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Rotate camera, take screenshots
    for i in range(72):
        angle = i * 5
        radians = np.radians(angle)
        camera.SetPosition(2 * np.sin(radians), 2 * np.cos(radians), 0)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        camera.ParallelProjectionOn()
        renderer.ResetCameraClippingRange()
        render_window.Render()

        # Take screenshot
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.SetFileName(f"{directory}/screenshot_{angle:03d}.png")
        writer.Write()

    render_window_interactor.Start()



def grid_fitting_non_interpolate(*args, **kwargs):
    if args and isinstance(args[0], dict):
        # If the first argument is a dictionary, extract all values from it
        coords = args[0]
        x2d_full = coords.get('x2d_full', None)
        y2d_full = coords.get('y2d_full', None)
        full_latency = coords.get('full_latency', None)
        full_latitudes = coords.get('full_latitudes', None)
        full_longitudes = coords.get('full_longitudes', None)
        latitudes = coords.get('latitudes', None)
        longitudes = coords.get('longitudes', None)
    else:
        # Otherwise, extract individual keyword arguments, defaulting to None if not provided
        x2d_full = kwargs.get('x2d_full', None)
        y2d_full = kwargs.get('y2d_full', None)
        full_latency = kwargs.get('full_latency', None)
        full_latitudes = kwargs.get('full_latitudes', None)
        full_longitudes = kwargs.get('full_longitudes', None)
        latitudes = kwargs.get('latitudes', None)
        longitudes = kwargs.get('longitudes', None)
    save = kwargs.pop('save', 1)
    file_address = kwargs.pop('file_name', 'contour.png')
    plot = kwargs.pop('plot', 0)
    colorbar = kwargs.pop('colorbar', 0)
    electrodes = kwargs.pop('electrodes', 1)

    # Create grid for interpolation
    grid_x_full, grid_y_full = np.mgrid[np.min(x2d_full):np.max(x2d_full):600j, np.min(y2d_full):np.max(y2d_full):300j]


    # Interpolate
    grid_z_full = griddata((x2d_full, y2d_full), full_latency, (grid_x_full, grid_y_full), method='cubic')

    # Plotting
    x2d, y2d = spherical_to_2d(latitudes,longitudes)

    # Example array
    array = np.array(grid_x_full[:,0])

    # Find indices where the condition is true
    indices = np.where((array > -160) & (array < 200))[0]

    grid_x_target = grid_x_full[indices]
    grid_y_target = grid_y_full[indices]
    grid_z_target = grid_z_full[indices]

    # Create a figure with a specific aspect ratio (2:1 width:height)
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed

    # Plot the data
    contour = ax.contourf(grid_x_target, grid_y_target, grid_z_target, levels=15, cmap='jet')
    if(electrodes):
        plt.scatter(x2d, y2d, c='red', edgecolor='k') 


    if(save):
        # Turn off the axis
        ax.axis('off')
        # Adjust layout
        plt.tight_layout(pad=0)

        # Ensure the saved plot has the same aspect ratio as specified
        fig.savefig(file_address, bbox_inches='tight', pad_inches=0, dpi=300)  # High resolution

        plt.close(fig)  # Close the plot
    elif(plot):
        if(colorbar):
            plt.colorbar()
            plt.show()
    grid_variables = {
    "grid_x_full": grid_x_full,
    "grid_y_full": grid_y_full,
    "grid_z_full": grid_z_full,
    "grid_x_target": grid_x_target,
    "grid_y_target": grid_y_target,
    "grid_z_target": grid_z_target
    }

    return grid_variables


def create_vector_field_sphere(grid_x_target, grid_y_target, grid_z_target, r = 250, scale=20, step=3, ratio=4, x2d=None, y2d=None, save=0, file_name='vector_plot.png'):
    rad_to_micrometers = 0.01745329251
    spacing = rad_to_micrometers * r  # Convert each step in grid to micrometers
    #Obtain the slowness function
    U, V = np.gradient(grid_z_target, spacing, spacing, edge_order=2)

    # Compute the magnitude of the gradient vectors
    magnitude = np.sqrt(U**2 + V**2)
    # Invert the magnitude to interpret as speed, handling divisions by zero
    #Convert it to CV
    U = U/magnitude**2
    V = V/magnitude**2

    #Calculate the magnitude for speed
    magnitude = 1/magnitude

    # Sample the grid, vectors, and magnitude for a sparser representation
    sparse_grid_x_target = grid_x_target[::step, ::ratio*step]
    sparse_grid_y_target = grid_y_target[::step, ::ratio*step]
    sparse_U = U[::step, ::ratio*step]
    sparse_V = V[::step, ::ratio*step]
    sparse_magnitude = magnitude[::step, ::ratio*step]

    # Create a figure with a specific aspect ratio (2:1 width:height)
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed

    # Plot the data
    contour = ax.contourf(grid_x_target, grid_y_target, grid_z_target, levels=15, cmap='jet')
    if x2d is not None:
        plt.scatter(x2d, y2d, c='red', edgecolor='k')

    # Use the magnitude as color for the quiver plot
    quiver = plt.quiver(sparse_grid_x_target, sparse_grid_y_target, sparse_U, sparse_V, sparse_magnitude, scale=scale, cmap='inferno')

    # Add a colorbar to represent the magnitude of the vectors
    plt.colorbar(quiver, ax=ax, label='Conduction Velocity (mm/s)')

    # Turn off the axis if save is True
    if save:
        ax.axis('off')

        # Adjust layout
        plt.tight_layout(pad=0)

        # Ensure the saved plot has the same aspect ratio as specified
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=600)  # High resolution

        plt.close(fig)  # Close the plot
    else:
        plt.show()

    return

def convert_degrees_to_radians(degrees):
    radians = degrees * (math.pi / 180)
    return radians

def create_vector_field(grid_x_target, grid_y_target, grid_z_target, r = 250, max_speed = 200,scale=20, step=3, ratio=1, x2d=None, y2d=None, save=0, file_name='vector_plot.png', plot = 1):
    print(grid_z_target.shape)
    (x_grid_count,y_grid_count) = grid_z_target.shape
    rad_y = 180/y_grid_count
    rad_x = 360/x_grid_count

    rad_to_micrometers = 0.01745329251
    grid_x_to_micrometers = rad_x*rad_to_micrometers*r
    grid_y_to_micrometers = rad_y*rad_to_micrometers*r
    U, V = np.gradient(grid_z_target[::step, ::ratio*step], 
                       grid_x_to_micrometers*step,
                        grid_y_to_micrometers*ratio*step,
                        edge_order=2)
    
    cos_y = np.cos(np.deg2rad(grid_y_target[::step, ::ratio*step]))

    # Adjust U based on cos(y)
    #U = U * cos_y
    
    # Compute the magnitude of the gradient vectors
    magnitude = np.sqrt(U**2 + V**2)
    # Invert the magnitude to interpret as speed, handling divisions by zero
    U = U/magnitude**2 * cos_y
    V = V/magnitude**2
    magnitude = 1/magnitude
    # print(magnitude)
    magnitude = np.where(magnitude > max_speed,max_speed, magnitude)

    # Sample the grid, vectors, and magnitude for a sparser representation
    sparse_grid_x_target = grid_x_target[::step, ::ratio*step]
    sparse_grid_y_target = grid_y_target[::step, ::ratio*step]
    # sparse_U = U[::step, ::ratio*step]
    # sparse_V = V[::step, ::ratio*step]
    sparse_U = U
    sparse_V = V
    sparse_magnitude = magnitude#[::step, ::ratio*step]

    # Create a figure with a specific aspect ratio (2:1 width:height)
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed

    # Plot the data
    contour = ax.contourf(grid_x_target, grid_y_target, grid_z_target, levels=15, cmap='jet')
    plt.colorbar(contour, ax=ax, label='Latency (ms)')
    if x2d is not None:
        plt.scatter(x2d, y2d, c='red', edgecolor='k')

    # Use the magnitude as color for the quiver plot
    log_magnitude = np.log(sparse_magnitude + 10)  # Adding a small value to avoid log(0)

    # Normalize log_magnitude to scale U and V
    norm_log_magnitude = log_magnitude / np.linalg.norm(log_magnitude)

    # Adjust U and V based on log magnitude for vector length
    adjusted_U = sparse_U  *norm_log_magnitude/magnitude
    adjusted_V = sparse_V *norm_log_magnitude/magnitude

    # Create the quiver plot with adjusted U and V
    quiver = plt.quiver(sparse_grid_x_target[1:-1, 1:-1], 
                        sparse_grid_y_target[1:-1, 1:-1], 
                        adjusted_U[1:-1, 1:-1], adjusted_V[1:-1, 1:-1], 
                        sparse_magnitude[1:-1, 1:-1], scale=scale, 
                        cmap='viridis')

    # Add a colorbar to represent the original magnitude of the vectors
    plt.colorbar(quiver, label='Conduction Velocity (mm/s)')
    plt.axis('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    # Turn off the axis if save is True
    if save:
        ax.axis('off')

        # Adjust layout
        plt.tight_layout(pad=0)

        # Ensure the saved plot has the same aspect ratio as specified
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)  # High resolution

        plt.close(fig)  # Close the plot
    elif(plot):
        plt.show()
    else:
        plt.close()

    return sparse_magnitude

def rotate_spherical_coords(long, lat, phi, axis = 'x'):
    lat_rad = np.radians(lat)
    long_rad = np.radians(long)
    x = np.cos(lat_rad) * np.cos(long_rad)
    y = np.cos(lat_rad) * np.sin(long_rad)
    z = np.sin(lat_rad)
    theta_radians = np.radians(phi)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_radians), -np.sin(theta_radians)],
                    [0, np.sin(theta_radians), np.cos(theta_radians)]])
    if axis != 'x':
        R_x = np.array([[np.cos(theta_radians), 0, np.sin(theta_radians)],
                    [0, 1, 0],
                    [-np.sin(theta_radians), 0, np.cos(theta_radians)]])

    coordinates = np.vstack((x, y, z))
    rotated_coordinates = R_x @ coordinates
    x_rotated, y_rotated, z_rotated = rotated_coordinates
    lat_rotated_rad = np.arcsin(z_rotated)
    long_rotated_rad = np.arctan2(y_rotated, x_rotated)
    lat_rotated_deg = np.degrees(lat_rotated_rad)
    long_rotated_deg = np.degrees(long_rotated_rad)
    return long_rotated_deg, lat_rotated_deg


    

from scipy.spatial import KDTree

def center_point_to_00(center_x, center_y, grid_x, grid_y, grid_z):
    #print(grid_x.shape)
    # Step 1: Find the row index closest to center_x (since rows have constant value across them)
    row_diff = np.abs(grid_x[:, 0] - center_x)  # Compute difference for one column, since all columns are the same for a row
    closest_row_index = np.argmin(row_diff)  # Index of the closest row

    # Step 2: Calculate shift amount to move closest row to the middle
    num_rows = grid_x.shape[0]
    middle_row_index = num_rows // 2  # Integer division to get the middle row index
    shift_amount = middle_row_index - closest_row_index  # Positive if need to shift down, negative if up

    # Step 3: Shift grid_z based on calculated shift_amount
    grid_z = np.roll(grid_z, shift_amount, axis=0)  # Adjust shifting to be vertical

    phi = center_y
    # Step 1: Transform all points
    flat_grid_x, flat_grid_y = grid_x.flatten(), grid_y.flatten()
    rotated_long, rotated_lat = rotate_spherical_coords(flat_grid_x, flat_grid_y, phi, axis='y')
    rotated_grid_x = rotated_long.reshape(grid_x.shape)
    rotated_grid_y = rotated_lat.reshape(grid_y.shape)

    # Step 2: Create a new grid and assign values
    # Here, we directly use the rotated grid as our new grid for simplicity
    # For a real application, you might need to interpolate values based on the distances to the nearest neighbors
    x = np.linspace(-180, 180.1, 605)
    y = np.linspace(-90, 90.1, 300)
    grid_x_new, grid_y_new = np.meshgrid(x, y)
    grid_x_new = grid_x_new.T
    grid_y_new = grid_y_new.T# Flatten the rotated grids and the original grid_z for processing
    rotated_grid_x_flat = rotated_grid_x.flatten()
    rotated_grid_y_flat = rotated_grid_y.flatten()
    grid_z_flat = grid_z.flatten()


    # Combine the rotated coordinates into a single array for KDTree
    rotated_coords = np.vstack((rotated_grid_x_flat, rotated_grid_y_flat)).T

    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(rotated_coords)

    # Flatten the new grid coordinates
    grid_x_new_flat = grid_x_new.flatten()
    grid_y_new_flat = grid_y_new.flatten()
    new_coords = np.vstack((grid_x_new_flat, grid_y_new_flat)).T

    # Find the nearest neighbors in the rotated grid for each point in the new grid
    distances, indices = tree.query(new_coords)

    # Assign the z values from the nearest rotated grid points to the new grid
    grid_z_new_flat = grid_z_flat[indices]

    # Reshape the flat grid_z_new back to the original grid shape
    grid_z_new = grid_z_new_flat.reshape(grid_x_new.shape)


    return grid_x_new, grid_y_new, grid_z_new


def create_vector_by_line(grid_x_target, grid_y_target, grid_z_target, r = 250, max_speed = 200,scale=20, step=3, ratio=1, x2d=None, y2d=None, save=0, file_name='vector_plot.png', plot = 1):
    (x_grid_count,y_grid_count) = grid_z_target.shape
    rad_y = 180/y_grid_count
    rad_x = 360/x_grid_count

    rad_to_micrometers = 0.01745329251
    grid_x_to_micrometers = rad_x*rad_to_micrometers*r
    grid_y_to_micrometers = rad_y*rad_to_micrometers*r
    #print(grid_x_to_micrometers,grid_y_to_micrometers)
    U, V = np.gradient(grid_z_target[::step, ::ratio*step], 
                       grid_x_to_micrometers*ratio*step,
                       grid_y_to_micrometers*step,
                        edge_order=2)
    # print("Shape of grid z:", grid_z_target[::step, ::ratio*step].shape)
    # print("Shape of U: ", U.shape)

    # Compute the magnitude of the gradient vectors
    magnitude = np.sqrt(U**2 + V**2)+0.001
    # Invert the magnitude to interpret as speed, handling divisions by zero
    U = U/magnitude**2 
    V = V/magnitude**2
    magnitude = 1/magnitude
    # print(magnitude)
    magnitude = np.where(magnitude > max_speed,max_speed, magnitude)



    return U, V


def create_vector_field_locally(grid_x_target, grid_y_target, 
                                grid_z_target, r = 250, max_speed = 200,scale=20, 
                                step=3, ratio=1, x2d=None, y2d=None, save=0, file_name='vector_plot.png', 
                                plot = 1, plot_each = False, print_progress = False):
    (y_grid_count,x_grid_count) = grid_z_target.shape
    rad_y = 180/y_grid_count
    rad_x = 360/x_grid_count

    rad_to_micrometers = 0.01745329251
    grid_x_to_micrometers = rad_x*rad_to_micrometers*r
    grid_y_to_micrometers = rad_y*rad_to_micrometers*r
    U, V = np.gradient(grid_z_target[::step, ::ratio*step], 
                        grid_y_to_micrometers*step,
                        grid_x_to_micrometers*ratio*step,
                        edge_order=2)
    print(U.shape)

    for i in range(grid_z_target[::step, ::ratio*step].shape[0]):
        for j in range(grid_z_target[::step, ::ratio*step].shape[1]):
            if(print_progress):
                if (grid_z_target[::step, ::ratio*step].shape[1]*i+j+1)%100 ==0:
                    print("Progress: ", 
                  str(grid_z_target[::step, ::ratio*step].shape[1]*i+j+1),
                  "/",str(grid_z_target[::step, ::ratio*step].shape[0]*grid_z_target[::step, ::ratio*step].shape[1]))
            center_x = grid_x_target[::step, ::ratio*step][i,j]
            center_y = grid_y_target[::step, ::ratio*step][i,j]
            
            #print(center_x)
            grid_x_new, grid_y_new, grid_z_new = center_point_to_00(center_x,center_y,grid_x_target, grid_y_target,grid_z_target)
            if(plot_each):
                # Create a figure with a specific aspect ratio (2:1 width:height)
                fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed
                # Plot the data
                contour = ax.contourf(grid_x_new, grid_y_new, grid_z_new, levels=15, cmap='jet')
                plt.colorbar(contour, ax=ax, label='Latency (ms)')
                if x2d is not None:
                    plt.scatter(x2d, y2d, c='red', edgecolor='k')
                plt.show()
                plt.matshow(np.sqrt(U**2+V**2).T)
                plt.show()
            U_line,V_line = create_vector_by_line(grid_x_new,
                                grid_y_new,
                                grid_z_new, 
                                max_speed= 400,
                                step = 10,
                                scale = 2,
                                )#[1:-1, 1:-1]#[6:-6, 6:-6] #[1:-1, 1:-1]
            #print(U_line[30,15], V_line[30,15], str(np.sqrt(U_line[30,15]**2+V_line[30,15]**2)))#,str(np.sqrt(U_line**2+V_line**2).max()))
            U[i,j],V[i,j] = U_line[30,15], V_line[30,15]
    
    magnitude = np.sqrt(U**2 + V**2)
    # print(magnitude)
    magnitude = np.where(magnitude > max_speed,max_speed, magnitude)

    # Sample the grid, vectors, and magnitude for a sparser representation
    sparse_grid_x_target = grid_x_target[::step, ::ratio*step]/10
    sparse_grid_y_target = grid_y_target[::step, ::ratio*step]/10
    # sparse_U = U[::step, ::ratio*step]
    # sparse_V = V[::step, ::ratio*step]
    sparse_U = U
    sparse_V = V
    sparse_magnitude = magnitude#[::step, ::ratio*step]

    # Create a figure with a specific aspect ratio (2:1 width:height)
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed

    # Plot the data
    contour = ax.contourf(grid_x_target, grid_y_target, grid_z_target/10, levels=15, cmap='jet')
    plt.colorbar(contour, ax=ax, label='Latency (ms)')
    if x2d is not None:
        plt.scatter(x2d, y2d, c='red', edgecolor='k')

    # Use the magnitude as color for the quiver plot
    log_magnitude = np.log(sparse_magnitude + 0.1)  # Adding a small value to avoid log(0)

    # Normalize log_magnitude to scale U and V
    norm_log_magnitude = log_magnitude / np.linalg.norm(log_magnitude)

    # Adjust U and V based on log magnitude for vector length
    adjusted_U = sparse_U  *norm_log_magnitude/magnitude
    adjusted_V = sparse_V *norm_log_magnitude/magnitude
    
    # Create the quiver plot with adjusted U and V
    quiver = plt.quiver(sparse_grid_x_target, 
                        sparse_grid_y_target, 
                        adjusted_U, adjusted_V, 
                        sparse_magnitude, scale=scale, 
                        cmap='viridis')

    # Add a colorbar to represent the original magnitude of the vectors
    plt.colorbar(quiver, label='Conduction Velocity (cm/s)')
    plt.axis('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    # Turn off the axis if save is True
    if save:
        ax.axis('off')

        # Adjust layout
        plt.tight_layout(pad=0)

        # Ensure the saved plot has the same aspect ratio as specified
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)  # High resolution

        plt.close(fig)  # Close the plot
    elif(plot):
        plt.show()
    else:
        plt.close()
    grids = {}
    grids['U'] = U
    grids['V'] = V
    grids['x'] = sparse_grid_x_target*10
    grids['y'] = sparse_grid_y_target*10
    return sparse_magnitude, grids
    
    

def rescaled_coordinate(alpha = 1):
    # alpha = diameter of the device/diameter of the organoid
    latitudes = np.array([0, 60, -60, 0, 0, 60, -60, 0, 0, -60, 60, 0, 0, -60, 60, 0])
    longitudes = np.array([22.5, 0, 90, 67.5, 112.5, 90, 180, 157.5, -22.5, 0, -90, -67.5, -112.5, -90, 180, -157.5])
    longtitude_shifting = 22.5*(1-alpha)
    direction = np.array([-1, 0, 0, +1, -1, 0, 0, +1, +1, 0, 0, -1, +1, 0, 0, -1])
    changed_longtitudes = longitudes + longtitude_shifting*direction
    changed_latitudes = (latitudes+90)*alpha - 90

    return changed_longtitudes,changed_latitudes

def CVPlot(vec, grids,U, V, x,y,
           grid_x_target, grid_y_target, 
            grid_z_target, r = 250, max_speed = 200,scale=20, 
            step=3, ratio=1, x2d=None, y2d=None, save=0, file_name='vector_plot.png', 
            plot = 1, plot_each = False, print_progress = False):
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed
    contour = ax.contourf(grid_x_target, grid_y_target, grid_z_target, levels=15, cmap='jet')
    plt.colorbar(contour, ax=ax, label='Latency (ms)')
    if x2d is not None:
        plt.scatter(x2d, y2d, c='k', edgecolor='k')
    magnitude = np.sqrt(U**2 + V**2)

    log_magnitude = np.log(magnitude + 0.1)  # Adding a small value to avoid log(0)
    norm_log_magnitude = log_magnitude / np.linalg.norm(log_magnitude)
    adjusted_U = U  *norm_log_magnitude/magnitude
    adjusted_V = V *norm_log_magnitude/magnitude
    quiver = plt.quiver(x, 
                        y, 
                        adjusted_U, adjusted_V, 
                        magnitude/10, scale=scale, 
                        cmap='viridis')
    plt.colorbar(quiver, label='Conduction Velocity (cm/s)')
    plt.axis('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
     # Turn off the axis if save is True
    if save:
        ax.axis('off')

        # Adjust layout
        plt.tight_layout(pad=0)

        # Ensure the saved plot has the same aspect ratio as specified
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=300)  # High resolution

        plt.close(fig)  # Close the plot
    elif(plot):
        plt.show()
    else:
        plt.close()



import numpy as np

def Frbf(d, epsilon=300.0):
    return np.exp(-(d**2) / epsilon**2)

def calculate_distance_vector(longitudes, latitudes, target_lon, target_lat):
    """
    Calculate the distance matrix for arrays of longitudes and latitudes.
    """
    n = len(longitudes)
    vector = np.zeros((n))
    for i in range(n):
        vector[i] = haversine(longitudes[i], latitudes[i], target_lon, target_lat)
    return vector
# Function to interpolate at a new point (requires its distances to all known points)
def interpolate(new_distances, lambda_, epsilon=300.0):
    return np.dot(Frbf(new_distances, epsilon=epsilon), lambda_)


def grid_fitting_rbf_sphere(latency, longitudes,latitudes, distance_matrix,
        size_x = 605,  size_y = 300, save = 0, file_address = 'contour_sphere.png',electrodes = 1 , plot = 1,
        colorbar = 0, target_long = None, target_lat = None, divide_time_by = 1,
        dense = 1, color = 'black', epsilon=300.0):

    latency = latency/divide_time_by
    A = Frbf(distance_matrix,  epsilon=epsilon)
    lambda_ = np.linalg.solve(A, latency)


    if target_long is None:
        longitudes_grid = np.linspace(-180, 180, size_x)
        latitudes_grid = np.linspace(-90, 90, size_y)
        grid_x, grid_y = np.meshgrid(longitudes_grid, latitudes_grid)
        grid_x = grid_x.T
        grid_y = grid_y.T
        # Initialize grid_z with the same shape as grid_x and grid_y
        grid_z = np.zeros_like(grid_x)

        for i in range(grid_z.shape[0]):
            for j in range(grid_z.shape[1]):
                new_point_distances = calculate_distance_vector(
                    longitudes, latitudes, grid_x[i,j],grid_y[i,j])
                grid_z[i,j] = interpolate(new_point_distances, lambda_, epsilon=epsilon)/10
        

        fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed
            
        # Plot the data
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=15, cmap='jet')
        if(electrodes):
            plt.scatter(longitudes, latitudes, c=color, edgecolor='k') 


        if save:
 
            ax.axis('off')  # Ensures no axis information is included in the output image
            plt.tight_layout(pad=0)  # Reduces or removes padding around the plot
            # Save the figure with adjusted settings to eliminate white space
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(file_address, bbox_inches='tight', pad_inches=0, dpi=600)
            #plt.show()
            plt.close(fig)  # Close the plot to free up memory

        elif(plot):
            plt.axis('equal')
            plt.xlim([-180,180])
            plt.ylim([-90,90])
            
            #plt.tight_layout(pad=0)
            if(colorbar):
                colorbar = plt.colorbar(contour)
                colorbar.set_label('Latency (ms)')
                plt.show()
        if not plot:
            plt.close()

    else:
        grid_z = []
        for longitude, latitude in zip(target_long,target_lat):
            new_point_distances = calculate_distance_vector(
                    longitudes, latitudes, longitude,latitude)
            grid_z.append(interpolate(new_point_distances, lambda_))
        grid_z_array = np.array(grid_z)

        return grid_z_array

    grid_variables = {
    "grid_x_full": grid_x,
    "grid_y_full": grid_y,
    "grid_z_full": grid_z,
    "grid_x_target": grid_x,
    "grid_y_target": grid_y,
    "grid_z_target": grid_z
    }
        
    return grid_variables

def haversine(lon1, lat1, lon2, lat2, r = 250):
    """
    Vectorized calculation of the Haversine distance between two sets of points.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * r

def fibonacci_sphere(samples=1):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        # Convert Cartesian to spherical coordinates
        longitude = np.degrees(np.arctan2(z, x))
        latitude = np.degrees(np.arcsin(y))

        points.append((longitude, latitude))

    return points

import math

def geographic_new_coords(lon, lat, distance, bearing, radius=250):
    """
    Calculate new geographic coordinates after traveling a specific distance on a sphere.
    
    :param lon: Starting longitude in degrees
    :param lat: Starting latitude in degrees
    :param distance: Distance to travel in kilometers
    :param bearing: Bearing in degrees from true North
    :param radius: Radius of the sphere (default is Earth's average radius in kilometers)
    :return: Tuple of new longitude and latitude in degrees
    """
    # Convert angles from degrees to radians
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    bearing_rad = math.radians(bearing)
    
    # Convert travel distance to angular distance (in radians)
    angular_distance = distance / radius
    
    # Calculate the new latitude (in radians)
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(angular_distance) +
                            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad))
    
    # Calculate the new longitude (in radians)
    new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
                                       math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat_rad))
    
    # Convert the new coordinates back to degrees
    new_lon = math.degrees(new_lon_rad)
    new_lat = math.degrees(new_lat_rad)

    if bearing == 0:
        new_lat = lat + distance/radius*180/math.pi
    if bearing == 180:
        new_lat = lat - distance/radius*180/math.pi
    return new_lon, new_lat

def find_smallest_distance(points):
    if len(points) < 2:
        return float('inf')  # Return infinity if there are less than two points
    
    min_distance = float('inf')
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def create_vector_field_spherical_coords(latency, longitudes, latitudes, distance_matrix, samples=100, r = 250, deltaD = None, scale = 2000, distance_ratio = 2):
    points = fibonacci_sphere(samples=samples)
    smallest_distance = find_smallest_distance(points)
    #print(f"The smallest distance between any two points is {smallest_distance}")
    distance_on_sphere = smallest_distance*2*math.pi*r/360
    if deltaD is None:
        deltaD = distance_on_sphere/distance_ratio


    # Prepare lists to hold plot data
    lons = [point[0] for point in points]  # Longitude for each point
    lats = [point[1] for point in points]  # Latitude for each point
    us = []  # X components of unit vectors
    vs = []  # Y components of unit vectors

    for point in points:
        # Calculate the new coordinates towards the cardinal directions
        north_lon, north_lat = geographic_new_coords(point[0], point[1], deltaD, bearing=0, radius=r)
        south_lon, south_lat = geographic_new_coords(point[0], point[1], deltaD, bearing=180, radius=r)
        west_lon, west_lat = geographic_new_coords(point[0], point[1], deltaD, bearing=270, radius=r)
        east_lon, east_lat = geographic_new_coords(point[0], point[1], deltaD, bearing=90, radius=r)
        
        # Define target longitudes and latitudes for grid fitting
        target_long = [east_lon, west_lon, north_lon, south_lon]
        target_lat = [east_lat, west_lat, north_lat, south_lat]
        #print((target_lat[2]+target_lat[3])/2,target_lat)
        # Get grid sphere values for RBF fitting on the sphere
        grid_sphere = grid_fitting_rbf_sphere(latency, longitudes, latitudes, distance_matrix,
                                            plot=1, save=0, target_long=target_long, target_lat=target_lat)

        # Compute finite difference gradients
        gradient_x = (grid_sphere[0] - grid_sphere[1]) / (2*deltaD)
        gradient_y = (grid_sphere[2] - grid_sphere[3]) / (2*deltaD)

        # Normalize the gradient vectors
        gradient_norm = gradient_x**2 + gradient_y**2
        #print(1/np.sqrt(gradient_norm))
        cv_x = gradient_x / gradient_norm if gradient_norm >0.00001 else 0
        cv_y = gradient_y / gradient_norm if gradient_norm >0.00001 else 0

        # Append unit vector components to lists
        us.append(cv_x)
        vs.append(cv_y)

    # Plot the unit vectors as arrows
    plt.figure(figsize=(10, 6))
    plt.quiver(lons, lats, us, vs, scale=scale)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Gradient Direction Field')
    plt.grid(True)
    plt.show()
    vectors = {
        'us': us,
        'vs': vs,
        'points': points,
    }
    return vectors


import numpy as np
from scipy.interpolate import griddata

def normalize_coordinates(lon, lat):
    """ Normalize the longitude and latitude values to fit within the typical geographical ranges. """
    # Apply normalization logic for longitude
    lon = np.where(lon > 180, lon - 360, lon)
    lon = np.where(lon < -180, lon + 360, lon)

    # Apply normalization logic for latitude
    lat = np.where(lat > 90, 180 - lat, lat)
    lat = np.where(lat < -90, -180 - lat, lat)

    return lon, lat

def batch_interpolation(grid_x, grid_y, grid_z, target_lons, target_lats):
    """
    Interpolate to find the values at specific geographic coordinates.
    
    Parameters:
        grid_x (numpy.ndarray): The grid of longitude values.
        grid_y (numpy.ndarray): The grid of latitude values.
        grid_z (numpy.ndarray): The grid of associated data values (same shape as grid_x and grid_y).
        target_lons (list or array): List or array of target longitudes.
        target_lats (list or array): List or array of target latitudes.
    
    Returns:
        list: The interpolated values at the target locations.
    """
    # Normalize target coordinates
    target_lons, target_lats = normalize_coordinates(np.array(target_lons), np.array(target_lats))

    # Flatten the grids and the values for interpolation
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    values = grid_z.ravel()
    
    # Prepare target points for interpolation
    target_points = np.column_stack((target_lons, target_lats))
    
    # Interpolate the values
    interpolated_values = griddata(points, values, target_points, method='linear')
    
    return interpolated_values



def create_vector_field_spherical_coords_grids( grid_x, grid_y,grid_z, 
                samples=100, r = 250, deltaD = None,
                scale = 2000):
    points = fibonacci_sphere(samples=samples)
    smallest_distance = find_smallest_distance(points)
    #print(f"The smallest distance between any two points is {smallest_distance}")
    distance_on_sphere = smallest_distance*2*math.pi*r/360
    if deltaD is None:
        deltaD = distance_on_sphere/3


    # Prepare lists to hold plot data
    lons = [point[0] for point in points]  # Longitude for each point
    lats = [point[1] for point in points]  # Latitude for each point
    
    us, vs = [], []
    vec_n = []
    target_lons, target_lats = [], []

    for point in points:
        # Calculate new coordinates towards the cardinal directions
        bearings = [90, 270, 0, 180]  # East, West, North, South
        for bearing in bearings:
            lon, lat = geographic_new_coords(point[0], point[1], deltaD, bearing=bearing, radius=r)
            target_lons.append(lon)
            target_lats.append(lat)

    # Perform interpolation for all target points at once
    grid_values = batch_interpolation(grid_x, grid_y, grid_z, target_lons, target_lats)

    # Process gradients and normalize
    for i in range(0, len(grid_values), 4):
        east, west, north, south = grid_values[i], grid_values[i+1], grid_values[i+2], grid_values[i+3]
        gradient_x = (east - west) / (2 * deltaD)
        gradient_y = (north - south) / (2 * deltaD)
        gradient_norm = (gradient_x**2 + gradient_y**2)
        cv_x = gradient_x / gradient_norm if gradient_norm > 0.00001 else 0
        cv_y = gradient_y / gradient_norm if gradient_norm > 0.00001 else 0
        us.append(cv_x)
        vs.append(cv_y)
        vec_n.append(np.log(10+np.sqrt(cv_x**2 + cv_y**2)))


    

    # Plot the unit vectors as arrows
    plt.figure(figsize=(10, 6))
    quiver = plt.quiver(lons, lats, us, vs, vec_n, scale=scale,cmap='viridis')
    plt.colorbar(quiver, label='Conduction Velocity (cm/s)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Gradient Direction Field')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    vectors = {
        'us': us,
        'vs': vs,
        'points': points,
    }
    return vectors


import numpy as np
import vtk


def create_sphere(renderer, radius=5.0, resolution=100):
    """Creates a smooth sphere and adds it to the specified renderer."""
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(radius)
    sphere_source.SetPhiResolution(resolution//2)
    sphere_source.SetThetaResolution(resolution)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    renderer.AddActor(sphere_actor)

# Definitions of your functions here
def lat_lon_to_cartesian(lat, lon, radius=6):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])

def north_vector(lat, lon, radius=6):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dx_dlat = -radius * np.sin(lat_rad) * np.cos(lon_rad)
    dy_dlat = -radius * np.sin(lat_rad) * np.sin(lon_rad)
    dz_dlat = radius * np.cos(lat_rad)
    v = np.array([dx_dlat, dy_dlat, dz_dlat])
    return v / np.linalg.norm(v)

def east_vector(norm, north):
    east = np.cross(north, norm)
    return east / np.linalg.norm(east)


def add_arrow(renderer, position, direction, scale, color):
    """Adds a colored arrow to the specified renderer, oriented to point in the given direction."""
    arrow_source = vtk.vtkArrowSource()

    # Normalize the direction vector
    direction = np.array(direction)
    if np.linalg.norm(direction) == 0:
        return  # Avoid division by zero
    direction = direction / np.linalg.norm(direction)

    # Create the transformation to align the arrow
    transform = vtk.vtkTransform()
    z_axis = np.array([1, 0, 0])
    if np.allclose(direction, z_axis):
        pass  # No transformation needed if direction is already along z_axis
    elif np.allclose(direction, -z_axis):
        transform.RotateWXYZ(180, 1, 0, 0)  # Rotate 180 degrees around the x-axis
    else:
        # Compute the rotation axis and angle
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.degrees(np.arctan2(np.linalg.norm(rotation_axis), np.dot(z_axis, direction)))
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        transform.RotateWXYZ(rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])

    # Apply the transformation to the arrow source
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(arrow_source.GetOutputPort())
    transform_filter.Update()

    # Set up the mapper and actor
    arrow_mapper = vtk.vtkPolyDataMapper()
    arrow_mapper.SetInputConnection(transform_filter.GetOutputPort())

    arrow_actor = vtk.vtkActor()
    arrow_actor.SetMapper(arrow_mapper)
    arrow_actor.SetPosition(position)
    arrow_actor.SetScale(scale)
    arrow_actor.GetProperty().SetColor(color)

    renderer.AddActor(arrow_actor)

# Example use case remains the same as previously defined, utilizing this new add_arrow function.

def jet_colormap(value):
    """ Convert a number in the range 0 to 1 into a color using a jet colormap approximation. """
    four_value = 4 * value
    red = min(four_value - 1.5, -four_value + 4.5)
    green = min(four_value - 0.5, -four_value + 3.5)
    blue = min(four_value + 0.5, -four_value + 2.5)
    return (max(min(red, 1), 0), max(min(green, 1), 0), max(min(blue, 1), 0))

def viridis_colormap(value):
    """ Convert a number in the range 0 to 1 into a color using a viridis colormap approximation. """
    c = [[0.267004, 0.004874, 0.329415],
         [0.281412, 0.155834, 0.469201],
         [0.244972, 0.287675, 0.53726],
         [0.190631, 0.407061, 0.556089],
         [0.147607, 0.511733, 0.557049],
         [0.119699, 0.61849, 0.536347],
         [0.20803,  0.718701, 0.472873],
         [0.430983, 0.808473, 0.346476],
         [0.709898, 0.868751, 0.169257]]
    idx = int(value * (len(c) - 1))
    return tuple(c[idx])


def setup_scene_arrow(points, us, vs,scale = 0.04, rotate_x = 0, rotate_y = 0, rotate_z = 200,radius = 0.8, max_cv = None, min_cv = None):
    """Sets up the renderer and render window, and adds arrows based on given points and direction vectors."""
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 1200) 
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    amplitude_all = []
    color = (1, 0, 0)  # Red arrows
    create_sphere(renderer, radius=radius, resolution=100, 
                  rotate_x= rotate_x,rotate_y= rotate_y,rotate_z= rotate_z)
    max_amp,min_amp = 0,10000
    for idx, (lon, lat) in enumerate(points):
        position = lat_lon_to_cartesian(lat, lon, radius)
        #print(lon,lat,us[idx],vs[idx] )
        north = north_vector(lat, lon, radius)
        east = east_vector(position, north)
        direction = us[idx] * east + vs[idx] * north
        amplitude = np.sqrt(np.sum(us[idx]**2 + vs[idx]**2))
        if(amplitude<min_amp):
            min_amp = amplitude
        if(amplitude>max_amp):
            max_amp = amplitude
        #print(min_amp,max_amp)
        #print(amplitude)
        amplitude_all.append(amplitude)
        if(max_cv is None):
            normalized_intensity = (amplitude - 10) / (100 - 10)
        else:
            normalized_intensity = (amplitude - min_cv) / (max_cv - min_cv)
        normalized_intensity = np.clip(normalized_intensity, 0, 1)  # Ensure within bounds
        color = jet_colormap(normalized_intensity)

        direction = direction / np.linalg.norm(direction)  # Ensure the direction is a unit vector
        add_arrow(renderer, position, direction, scale*np.log(np.log(amplitude)+5), color)
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetPosition(10, 0, 0)  # Position the camera away from the origin along the Z-axis
    camera.SetFocalPoint(0, 0, 0)  # The camera looks towards the origin
    camera.SetViewUp(0, 0, 1)  # The Y-axis is up
    renderer.SetBackground(1, 1, 1)
    # create_sphere(renderer, radius=6.0, resolution=1000, 
    #               rotate_x= rotate_x,rotate_y= rotate_y,rotate_z= rotate_z)

    render_window_interactor.Start()
    #return amplitude_all


import vtk

def create_sphere(renderer, radius=5.0, resolution=100, texture_file="contour.png", rotate_x = 80
                  , rotate_y = 0, rotate_z = 0):
    """Creates a smooth sphere, maps a texture onto it, and adds it to the specified renderer."""
    # Create the sphere geometry
    image_path = texture_file  # Example path, adjust as necessary
    img = Image.open(image_path).convert('RGB')
    img_data = np.flipud(np.array(img))

    # Convert image data to a VTK compatible format
    height, width, _ = img_data.shape
    img_data_2d = img_data.reshape(height*width, 3)
    vtk_array = numpy_to_vtk(num_array=img_data_2d, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    # Create a VTK image and texture
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, 1)
    vtk_image.GetPointData().SetScalars(vtk_array)
    texture = vtk.vtkTexture()
    texture.SetInputData(vtk_image)
    texture.InterpolateOn()

    # Create the geometry and actor to apply the texture
    sphere = vtk.vtkTexturedSphereSource()
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)
    sphere.SetRadius(radius)
    sphere_mapper = vtk.vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere.GetOutputPort())
    sphere_actor = vtk.vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.SetTexture(texture)
    sphere_actor.RotateX(rotate_x)
    sphere_actor.RotateY(rotate_y)
    sphere_actor.RotateZ(rotate_z)
    renderer.AddActor(sphere_actor)



import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import os
import vtk
import numpy as np

def setup_scene_and_capture(points, us, vs, scale=0.04,rotate_x = 0, rotate_y = 0, rotate_z = 200, radius=0.8, max_cv=None, min_cv=None, 
                            CV_threshold = 80 , #This is for JKCO6
                            projection = False,ViewingFromTop = False, directory = "sphere_vector", texture_file="contour.png", auto_close = True):
    """Sets up a 3D visualization scene with arrows on a sphere, captures rotating views, and saves them as images.

    Args:
        points (list): List of (longitude, latitude) tuples representing points on the sphere.
        us (list): List of x-components of the vectors at each point.
        vs (list): List of y-components of the vectors at each point.
        scale (float, optional): Scaling factor for the arrows. Defaults to 0.04.
        rotate_x (int, optional): Rotation angle around x-axis in degrees. Defaults to 0.
        rotate_y (int, optional): Rotation angle around y-axis in degrees. Defaults to 0.
        rotate_z (int, optional): Rotation angle around z-axis in degrees. Defaults to 200.
        radius (float, optional): Radius of the sphere. Defaults to 0.8.
        max_cv (float, optional): Maximum conduction velocity for color normalization. Defaults to None.
        min_cv (float, optional): Minimum conduction velocity for color normalization. Defaults to None.
        CV_threshold (int, optional): Threshold for conduction velocity visualization. Defaults to 80.
        projection (bool, optional): Whether to use projection mode. Defaults to False.
        ViewingFromTop (bool, optional): Whether to view the sphere from top. Defaults to False.
        directory (str, optional): Directory to save captured images. Defaults to "sphere_vector".
        texture_file (str, optional): Path to texture file for the sphere. Defaults to "contour.png".
        auto_close (bool, optional): Whether to automatically close the window after capture. Defaults to True.

    Returns:
        None

    Note:
        - Creates 36 images rotating the view by 10 degrees each time
        - Images are saved as 'angle_[0-35].png' in the specified directory
        - Uses VTK for rendering and PIL for image processing
        - Arrow colors are determined by vector magnitude using jet colormap
    """
    # Create renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 1200)  # Larger window size
    render_window.AddRenderer(renderer)
    renderer.SetBackground(1, 1, 1)  # Set the background to white
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Setup camera
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    if(ViewingFromTop):
        camera.SetPosition(0, 0, 5)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(-1, 0, 0)


    camera.SetPosition(5, 0, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    if(projection):
        renderer.AutomaticLightCreationOff()
        renderer.UseShadowsOff()
        renderer.SetAmbient(1.0, 1.0, 1.0)

    # Render objects
    create_sphere(renderer, radius=radius, resolution=100, rotate_x=rotate_x, rotate_y=rotate_y, rotate_z=rotate_z, texture_file=texture_file)
    for idx, (lon, lat) in enumerate(points):
        position = lat_lon_to_cartesian(lat, lon, radius)
        north = north_vector(lat, lon, radius)
        east = east_vector(position, north)
        direction = us[idx] * east + vs[idx] * north
        amplitude = np.sqrt(np.sum(us[idx]**2 + vs[idx]**2))
       
        if CV_threshold:
            amplitude = min(amplitude, CV_threshold)
            
            color_intensity = amplitude / CV_threshold
            color_intensity = np.clip(color_intensity, 0, 1)
            color = jet_colormap(color_intensity)  
        else:
            normalized_intensity = (amplitude - min_cv) / (max_cv - min_cv) if max_cv and min_cv else (amplitude - 10) / (90)
            normalized_intensity = np.clip(normalized_intensity, 0, 1)
            color = jet_colormap(normalized_intensity)
        direction = direction / np.linalg.norm(direction)
        add_arrow(renderer, position, direction, scale * np.log(np.log(amplitude) + 5), color)

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize Window to Image Filter
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetScale(1)  # Image quality
    w2if.SetInputBufferTypeToRGBA()
    
    # Initialize the writer
    writer = vtk.vtkPNGWriter()

    # Rotate and capture images
    for i in range(36):  # 360 degrees in steps of 10 degrees
        camera.Azimuth(10)  # Rotate camera
        render_window.Render()  # Render the scene
        w2if.Modified()  # Update the filter to reflect new scene
        w2if.Update()  # Ensure the filter processes the latest image
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.SetFileName(f"{directory}/angle_{i}.png")
        writer.Write()


    # Finalize and close the window automatically
    if(auto_close):
        render_window.Finalize()
        render_window_interactor.TerminateApp()
    else:
        render_window_interactor.Start()



def calculate_distance_vector(longitudes, latitudes, target_lon, target_lat):
    """
    Calculate the distance matrix for arrays of longitudes and latitudes.
    """
    n = len(longitudes)
    vector = np.zeros((n))
    for i in range(n):
        vector[i] = haversine(longitudes[i], latitudes[i], target_lon, target_lat)
    return vector


# #Update:
# def Frbf(d, epsilon, method='gaussian'):
#     if method == 'gaussian':
#         return np.exp(-(d**2) / epsilon**2)
#     elif method == 'multiquadric':
#         return np.sqrt((d/epsilon)**2 + 1)
#     elif method == 'inverse_quadratic':
#         return 1 / (1 + (d/epsilon)**2)
#     elif method == 'inverse_multiquadric':
#         return 1 / np.sqrt((d/epsilon)**2 + 1)
#     elif method == 'thin_plate':
#         return d**2 * np.log(d + 1e-10)  # add small constant to avoid log(0)
#     elif method == 'cubic':
#         return d**3
#     elif method == 'quintic':
#         return d**5
#     elif method == 'linear':
#         return d
#     else:
#         raise ValueError("Unsupported method. Choose from 'gaussian', 'multiquadric', 'inverse_quadratic', 'inverse_multiquadric', 'thin_plate', 'cubic', 'quintic', or 'linear'.")


# def interpolate(new_distances, lambda_, epsilon, method='gaussian'):
#     return np.dot(Frbf(new_distances, epsilon=epsilon, method=method), lambda_)


# def grid_fitting_rbf_sphere(latency, longitudes, latitudes, distance_matrix,
#         size_x=605, size_y=300, save=0, file_address='contour_sphere.png', electrodes=1, plot=1,
#         colorbar=0, target_long=None, target_lat=None, divide_time_by=1,
#         dense=1, color='black', epsilon=300.0, method='gaussian',levels=np.linspace(0, 50, 100)):

#     latency = latency/divide_time_by
#     A = Frbf(distance_matrix, epsilon=epsilon, method=method)
#     lambda_ = np.linalg.solve(A, latency)

#     if target_long is None:
#         longitudes_grid = np.linspace(-180, 180, size_x)
#         latitudes_grid = np.linspace(-90, 90, size_y)
#         grid_x, grid_y = np.meshgrid(longitudes_grid, latitudes_grid)
#         grid_x = grid_x.T
#         grid_y = grid_y.T
#         grid_z = np.zeros_like(grid_x)

#         for i in range(grid_z.shape[0]):
#             for j in range(grid_z.shape[1]):
#                 new_point_distances = calculate_distance_vector(
#                     longitudes, latitudes, grid_x[i,j], grid_y[i,j])
#                 grid_z[i,j] = interpolate(new_point_distances, lambda_, epsilon=epsilon, method=method)/10
        
#         fig, ax = plt.subplots(figsize=(8, 4))
        
#         contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='jet')
#         if electrodes:
#             plt.scatter(longitudes, latitudes, c=color, edgecolor='k') 

#         if save:
#             ax.axis('off')
#             plt.tight_layout(pad=0)
#             fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
#             fig.savefig(file_address, bbox_inches='tight', pad_inches=0, dpi=600)
#             plt.close(fig)
#         elif plot:
#             plt.axis('equal')
#             plt.xlim([-180,180])
#             plt.ylim([-90,90])
#             if colorbar:
#                 colorbar = plt.colorbar(contour)
#                 colorbar.set_label('Latency (ms)')
#             plt.show()
#         if not plot:
#             plt.close()

#     else:
#         grid_z = []
#         for longitude, latitude in zip(target_long, target_lat):
#             new_point_distances = calculate_distance_vector(
#                     longitudes, latitudes, longitude, latitude)
#             grid_z.append(interpolate(new_point_distances, lambda_, epsilon=epsilon, method=method))
#         grid_z_array = np.array(grid_z)

#         return grid_z_array

#     interpolated_at_original = np.dot(A, lambda_)
#     difference = np.abs(interpolated_at_original - latency)
#     #print("Differences:", difference)
    
#     grid_variables = {
#         "grid_x_full": grid_x,
#         "grid_y_full": grid_y,
#         "grid_z_full": grid_z,
#         "grid_x_target": grid_x,
#         "grid_y_target": grid_y,
#         "grid_z_target": grid_z
#     }
        
#     return grid_variables

