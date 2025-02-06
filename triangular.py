import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import seaborn as sns

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

def calculate_distance_matrix(longitudes, latitudes, r = 250):
    """
    Calculate the distance matrix for arrays of longitudes and latitudes.
    """
    n = len(longitudes)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = haversine(longitudes[i], latitudes[i], longitudes[j], latitudes[j], r = r)
    return distance_matrix

def calculate_velocities(distance_matrix, filtered_connected_indices, peaks):
    velocities = []  # List to hold arrays of velocities for each n_peaks
    start_indices = []  # List to hold arrays of start indices for each n_peaks
    end_indices = []  # List to hold arrays of end indices for each n_peaks
    
    n_peaks = peaks.shape[1]
    
    for n in range(n_peaks):
        peak_column = peaks[:, n]  # Activation times for the current column n
        velocities_n = []  # Velocities for the current column n
        start_indices_n = []  # Start indices for the current column n
        end_indices_n = []  # End indices for the current column n
        
        for j in range(len(filtered_connected_indices)):
            for k in filtered_connected_indices[j]:
                # Calculate velocity if the time difference is not zero to avoid division by zero
                time_difference = peak_column[k] - peak_column[j]
                if time_difference != 0:
                    velocity = distance_matrix[j, k] / time_difference/10 # um/ms = mm/s
                    velocities_n.append(velocity)
                    start_indices_n.append(j)
                    end_indices_n.append(k)
        
        velocities.append(np.array(velocities_n))
        start_indices.append(np.array(start_indices_n))
        end_indices.append(np.array(end_indices_n))
    
    return velocities, start_indices, end_indices


def calculate_conduction_velocity(points, latencies):
    """
    Calculate the conduction velocity given 3 points and their latencies.
    
    :param points: A numpy array of shape (3, 2) containing the coordinates of the three points.
    :param latencies: A numpy array of shape (3,) containing the latency at each point.
    :return: The calculated conduction velocity.
    """
    # Construct the design matrix A and vector b (T_i)
    A = np.hstack([np.ones((3, 1)), points])  # Add a column of ones for alpha_0
    b = latencies.reshape(-1, 1)
    
    # Solve for the coefficients (alpha_0, alpha_1, alpha_2)
    try:
        alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        # Use alpha as needed
    except np.linalg.LinAlgError as e:
        print("An error occurred:", e)
        # Handle the error, for example by setting alpha to None or a default value
        num_parameters = A.shape[1]  # Number of columns in A represents the number of parameters
        alpha = np.array(range(num_parameters))
        
    # Extract alpha_1 and alpha_2 to compute the gradient
    grad_T = alpha[1:]
    
    # Calculate the magnitude of the gradient
    grad_T_magnitude = np.linalg.norm(grad_T)
    
    # Calculate the conduction velocity
    CV = 1 / grad_T_magnitude
    
    return CV


def calculate_third_point_coordinates(d12, d13, d23):
    """
    Calculate coordinates of the third point in a triangle given distances
    between the points. The first point is at (0, 0), and the second point is
    on the positive x-axis.
    
    :param d12: Distance between the first and second points.
    :param d13: Distance between the first and third points.
    :param d23: Distance between the second and third points.
    :return: Coordinates of the three points as a tuple of tuples.
    """
    # Coordinates of the first two points are known
    P1 = (0, 0)
    P2 = (d12, 0)
    
    # Calculate coordinates of the third point
    x3 = (d13**2 - d23**2 + d12**2) / (2 * d12)
    y3 = np.sqrt(d13**2 - x3**2)
    
    P3 = (x3, y3)
    
    return P1, P2, P3

def calculate_CVs_for_triangles(distance_matrix, selected_triangles, peaks):
    """
    Calculate the conduction velocities for each triangle for each set of latencies.

    :param distance_matrix: A (m, m) array with distances between points.
    :param selected_triangles: An array of shape (n, 3) with indices of points forming each triangle.
    :param peaks: An array of shape (m, j) with latencies for each point and each set.
    :return: An array of shape (j, n) with calculated CVs.
    """
    m, j = peaks.shape
    n = selected_triangles.shape[0]
    CVs = np.zeros((j, n))
    valid_triangles_indices = []
    # Assuming a simplified way to infer coordinates, 
    # would need a real method for complex cases.
    # Here, we should have a function like calculate_third_point_coordinates but for the entire set.
    # For simplicity in this outline, we skip direct coordinate calculation.
    
    # Placeholder: Infer coordinates from distance matrix, complex and requires assumptions
    # coordinates = infer_coordinates_from_distances(distance_matrix)
    
    for j_index in range(j):  # Iterate over each set of latencies
        for tri_index, triangle in enumerate(selected_triangles):
            # Extract latencies for the current triangle and set
            latencies = peaks[triangle, j_index]

            # Assuming we have a method to get coordinates (simplified here)
            d12 = distance_matrix[triangle[0], triangle[1]]
            d13 = distance_matrix[triangle[0], triangle[2]]
            d23 = distance_matrix[triangle[1], triangle[2]]
            # Use placeholder coordinates for illustration
            coordinates = calculate_third_point_coordinates(d12, d13, d23)
            
            # Calculate CV
            CV = calculate_conduction_velocity(coordinates, latencies)
            if(latencies[2] == latencies[1] and latencies[0] == latencies[1] ):
                continue
            if CV>400:
                continue
            CVs[j_index, tri_index] = CV
            valid_triangles_indices.append(tri_index)
    selected_triangles_filtered = selected_triangles[valid_triangles_indices]
    return CVs , selected_triangles_filtered

def mirror_long_lat(longitudes, latitudes):
    longitudes_wrapped_left = longitudes - 360
    longitudes_wrapped_right = longitudes + 360
    lon_shift = 180
    longitudes_up = longitudes - 90
    # For latitude modification, it seems you want to mirror across the equator, which is a different operation.
    # So, I will proceed with the mirror across the equator for both longitude and latitude sets
    latitudes_mirrored_top = -latitudes + 180  
    latitudes_mirrored_down = -latitudes - 180

    # Combine all sets
    # Original + Modified Longitude + Mirrored Latitude (across equator)
    all_longitudes = np.concatenate([longitudes, longitudes - lon_shift, longitudes - lon_shift,
                                    longitudes_wrapped_left, longitudes_wrapped_left - lon_shift, longitudes_wrapped_left - lon_shift,
                                    longitudes_wrapped_right, longitudes_wrapped_right - lon_shift , longitudes_wrapped_right - lon_shift])

    all_latitudes = np.concatenate([latitudes, latitudes_mirrored_top, latitudes_mirrored_down, 
                                    latitudes, latitudes_mirrored_top, latitudes_mirrored_down, 
                                    latitudes, latitudes_mirrored_top, latitudes_mirrored_down, ])
    # Reflect across the equator by inverting latitudes and keeping longitudes the same
    # Note: Adding and subtracting 90 to latitudes to "mirror" at those bounds is not geographically meaningful
    # since latitudes range from -90 to 90, so here I'll demonstrate a simple inversion

    # Print the new sets for verification
    # print("All Longitudes:", all_longitudes)
    # print("All Latitudes:", all_latitudes)
    plt.scatter(all_longitudes, all_latitudes)
    plt.show()
    # Perform Delaunay triangulation
    points = np.vstack([all_longitudes, all_latitudes]).T # Stack longitude and latitude for triangulation
    tri = Delaunay(points)

    # Plotting
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    plt.figure(figsize=(10, 5)) 
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.xlim([-180,180])
    plt.ylim([-90,90])

    plt.show()
    return all_longitudes, all_latitudes, tri, points

def do_both_calculations(peaks,
                         latitudes = None, longitudes = None, target_indices = None):
    if longitudes is None:
        latitudes = np.array([0, 60, -60, 0, 0, 60, -60, 0, 0, -60, 60, 0, 0, -60, 60, 0])
        longitudes = np.array([22.5, 0, 90, 67.5, 112.5, 90, 180, 157.5, -22.5, 0, -90, -67.5, -112.5, -90, 180, -157.5])
    if target_indices is None:
        target_indices = [0, 1, 2, 3, 4, 5, 6, 7,8,9, 10,11,12, 13, 14, 15]
    longitudes = longitudes[target_indices]
    latitudes = latitudes[target_indices]
    distance_matrix = calculate_distance_matrix(longitudes, latitudes)
    # Perform Delaunay triangulation
    points = np.vstack([longitudes, latitudes]).T # Stack longitude and latitude for triangulation
    tri = Delaunay(points)

    # Plotting
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    all_longitudes, all_latitudes , tri, points= mirror_long_lat(longitudes, latitudes)
    connected_indices = [[] for _ in range(len(points))]

    for simplex in tri.simplices:
        for i in simplex:
            connected_indices[i].extend(simplex[simplex != i])

    # Remove duplicates and sort
    connected_indices = [sorted(set(indices)) for indices in connected_indices]
    len_ind = len(target_indices)
    adjusted_connected_indices = []

    for indices in connected_indices:
        adjusted_indices = [i - (i//len_ind)*len_ind if i >= len_ind else i for i in indices]
        adjusted_connected_indices.append(adjusted_indices)

    targetad_connected_indices = adjusted_connected_indices[:len_ind]

    filtered_connected_indices = []

    for index, indices in enumerate(targetad_connected_indices):
        filtered_indices = [i for i in indices if i > index]
        filtered_connected_indices.append(filtered_indices)
    velocities, start_indices, end_indices = calculate_velocities(distance_matrix, filtered_connected_indices, peaks)
    # Flatten the list of numpy arrays into a single numpy array
    all_velocities = np.concatenate(velocities)

    # Take the absolute values of the velocities
    abs_velocities = np.abs(all_velocities)

    # Plot the histogram
    sns.histplot(abs_velocities, bins=30, kde=True)
    plt.xlabel('Velocity (cm/s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Absolute Velocities')
    plt.show()
    print("Mean:", np.mean(abs_velocities))
    print("Std:", np.std(abs_velocities))

    points = np.vstack([all_longitudes, all_latitudes]).T

    # Perform Delaunay triangulation
    tri = Delaunay(points)

    # Find all triangles that include one of the indices from 0 to 11
    selected_triangles = []
    for i, simplex in enumerate(tri.simplices):
        if any(vertex in range(0, len_ind) for vertex in simplex):
            selected_triangles.append(simplex)

    # Convert to a numpy array for convenience
    selected_triangles = np.array(selected_triangles)

    # selected_triangles now contains all the triangles you're interested in
    print("Selected triangles (indices):")
    print(len(selected_triangles))
    selected_triangles = np.array(selected_triangles)

    # Modify the indices in the triangles
    selected_triangles = selected_triangles % len_ind

    CVs,triangles = calculate_CVs_for_triangles(distance_matrix, selected_triangles, peaks)
    # Take the absolute values of the velocities
    abs_CVs = (np.abs(CVs)/10).flatten()
    abs_CVs = abs_CVs[abs_CVs!= 0]
    # Plot the histogram
    plt.xlim([0,40])
    sns.histplot(abs_CVs, bins=30, kde=True)
    plt.xlabel('Velocity (cm/s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Absolute Velocities')
    plt.show()
    print("Mean:", np.mean(abs_CVs))
    print("Std:", np.std(abs_CVs))

    return velocities, start_indices, end_indices,CVs,triangles

def do_triangular_calculations(peaks,
                         latitudes = None, longitudes = None, target_indices = None):
    if longitudes is None:
        latitudes = np.array([0, 60, -60, 0, 0, 60, -60, 0, 0, -60, 60, 0, 0, -60, 60, 0])
        longitudes = np.array([22.5, 0, 90, 67.5, 112.5, 90, 180, 157.5, -22.5, 0, -90, -67.5, -112.5, -90, 180, -157.5])
    if target_indices is None:
        target_indices = [0, 1, 2, 3, 4, 5, 6, 7,8,9, 10,11,12, 13, 14, 15]
    longitudes = longitudes[target_indices]
    latitudes = latitudes[target_indices]
    distance_matrix = calculate_distance_matrix(longitudes, latitudes)
    # Perform Delaunay triangulation
    points = np.vstack([longitudes, latitudes]).T # Stack longitude and latitude for triangulation
    tri = Delaunay(points)

    # Plotting
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    all_longitudes, all_latitudes , tri, points= mirror_long_lat(longitudes, latitudes)
    connected_indices = [[] for _ in range(len(points))]

    for simplex in tri.simplices:
        for i in simplex:
            connected_indices[i].extend(simplex[simplex != i])

    # Remove duplicates and sort
    connected_indices = [sorted(set(indices)) for indices in connected_indices]
    len_ind = len(target_indices)
    adjusted_connected_indices = []

    for indices in connected_indices:
        adjusted_indices = [i - (i//len_ind)*len_ind if i >= len_ind else i for i in indices]
        adjusted_connected_indices.append(adjusted_indices)

    targetad_connected_indices = adjusted_connected_indices[:len_ind]

    filtered_connected_indices = []

    for index, indices in enumerate(targetad_connected_indices):
        filtered_indices = [i for i in indices if i > index]
        filtered_connected_indices.append(filtered_indices)
    velocities, start_indices, end_indices = calculate_velocities(distance_matrix, filtered_connected_indices, peaks)
    # Flatten the list of numpy arrays into a single numpy array
    all_velocities = np.concatenate(velocities)

    # Take the absolute values of the velocities
    abs_velocities = np.abs(all_velocities)

    # Plot the histogram
    # sns.histplot(abs_velocities, bins=30, kde=True)
    # plt.xlabel('Velocity (cm/s)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Absolute Velocities')
    # plt.show()
    # print("Mean:", np.mean(abs_velocities))
    # print("Std:", np.std(abs_velocities))

    points = np.vstack([all_longitudes, all_latitudes]).T

    # Perform Delaunay triangulation
    tri = Delaunay(points)

    # Find all triangles that include one of the indices from 0 to 11
    selected_triangles = []
    for i, simplex in enumerate(tri.simplices):
        if any(vertex in range(0, len_ind) for vertex in simplex):
            selected_triangles.append(simplex)

    # Convert to a numpy array for convenience
    selected_triangles = np.array(selected_triangles)

    # selected_triangles now contains all the triangles you're interested in
    print("Selected triangles (indices):")
    print(len(selected_triangles))
    selected_triangles = np.array(selected_triangles)

    # Modify the indices in the triangles
    selected_triangles = selected_triangles % len_ind

    CVs,triangles = calculate_CVs_for_triangles(distance_matrix, selected_triangles, peaks)
    print("Hi")
    # Take the absolute values of the velocities
    abs_CVs = (np.abs(CVs)/10).flatten()
    abs_CVs = abs_CVs[abs_CVs!= 0]
    # Plot the histogram
    plt.xlim([0,40])
    sns.histplot(abs_CVs, bins=30, kde=True)
    plt.xlabel('Velocity (cm/s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Absolute Velocities')
    plt.show()
    print("Mean:", np.mean(abs_CVs))
    print("Std:", np.std(abs_CVs))

    return velocities, start_indices, end_indices,CVs,triangles


