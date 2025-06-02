import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap # Import for custom colormap
from numba import jit
import gmpy2
import random

# Set a high precision for gmpy2 calculations
gmpy2.get_context().precision = 100 # Increased precision

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_axis_off()

WIDTH, HEIGHT = 800, 800 # Increased resolution for more pixels
MAX_ITER = 50

# Define interesting Julia set constants
JULIA_CONSTANTS = [
    (gmpy2.mpfr('-0.8'), gmpy2.mpfr('0.156')),
    (gmpy2.mpfr('0.285'), gmpy2.mpfr('0.01')),
    (gmpy2.mpfr('-0.70176'), gmpy2.mpfr('-0.3842')),
    (gmpy2.mpfr('-0.4'), gmpy2.mpfr('0.6')),
    (gmpy2.mpfr('0.3'), gmpy2.mpfr('0.5')),
    (gmpy2.mpfr('-0.7269'), gmpy2.mpfr('0.1889')), # Seahorse valley
    (gmpy2.mpfr('0.35'), gmpy2.mpfr('0.35')),
    (gmpy2.mpfr('-0.1'), gmpy2.mpfr('0.65')),
    (gmpy2.mpfr('-0.12'), gmpy2.mpfr('0.77')),
    (gmpy2.mpfr('-0.54'), gmpy2.mpfr('0.54'))
]

# Randomly choose between Mandelbrot (10% chance) and Julia set (90% chance)
fractal_type = random.choice(['mandelbrot'] + ['julia'] * 9)
c_julia_x, c_julia_y = None, None # Initialize for Julia set constant

if fractal_type == 'mandelbrot':
    zoom = gmpy2.mpfr('1.0')
    center_x = gmpy2.mpfr('-0.743643887037151')
    center_y = gmpy2.mpfr('0.13182590420533')
    print("Generating Mandelbrot set.")
else: # fractal_type == 'julia'
    zoom = gmpy2.mpfr('0.8') # Julia sets often look good with a slightly different initial zoom
    center_x = gmpy2.mpfr('0.0') # Julia sets are often centered around 0,0
    center_y = gmpy2.mpfr('0.0')
    c_julia_x, c_julia_y = random.choice(JULIA_CONSTANTS)
    print(f"Generating Julia set with c = {float(c_julia_x):.5f} + {float(c_julia_y):.5f}i")

target_x, target_y = center_x, center_y

# Create a custom colormap: 0 (inside) is black, 1 (outside) is white
colors = [(0, 0, 0), (1, 1, 1)] # Black to White
custom_cmap = LinearSegmentedColormap.from_list('BlackAndWhite', colors, N=2) # N=2 for binary colors

image = ax.imshow(np.zeros((HEIGHT, WIDTH)), extent=[-2, 1, -1.5, 1.5], cmap=custom_cmap, origin='lower')

@jit(nopython=True)
def mandelbrot(x, y, max_iter):
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iter):
        z = z*z + c
        if abs(z) >= 2:
            return 1.0 # Outside: white
    return 0.0 # Inside: black

@jit(nopython=True)
def julia(zx, zy, c_julia_x, c_julia_y, max_iter):
    c = complex(c_julia_x, c_julia_y)
    z = complex(zx, zy)
    for i in range(max_iter):
        z = z*z + c
        if abs(z) >= 2:
            return 1.0 # Outside: white
    return 0.0 # Inside: black

# generate_fractal will now handle gmpy2 objects for precision, then convert to float for numba
def generate_fractal(center_x_mp, center_y_mp, zoom_mp, width, height, max_iter, fractal_type, c_julia_x_mp, c_julia_y_mp):
    scale_mp = gmpy2.mpfr('1.5') / zoom_mp
    x_min_mp = center_x_mp - scale_mp
    x_max_mp = center_x_mp + scale_mp
    y_min_mp = center_y_mp - scale_mp
    y_max_mp = center_y_mp + scale_mp

    fractal = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            # Perform high-precision calculations for zx and zy
            zx_mp = x_min_mp + (gmpy2.mpfr(x) / gmpy2.mpfr(width)) * (x_max_mp - x_min_mp)
            zy_mp = y_min_mp + (gmpy2.mpfr(y) / gmpy2.mpfr(height)) * (y_max_mp - y_min_mp)
            
            # Convert to standard float for the numba-compiled functions
            if fractal_type == 'mandelbrot':
                fractal[y, x] = mandelbrot(float(zx_mp), float(zy_mp), max_iter)
            else: # julia
                fractal[y, x] = julia(float(zx_mp), float(zy_mp), float(c_julia_x_mp), float(c_julia_y_mp), max_iter)
    return fractal

def find_most_detailed_region(fractal):
    # Calculate gradient to find areas of change (edges between 0 and 1)
    grad = np.abs(np.gradient(fractal))
    detail = grad[0] + grad[1]

    # Find all points that are on the edge (where detail > 0)
    edge_y, edge_x = np.where(detail > 0)

    # If no edge points found (e.g., completely black or white image), return the center
    if len(edge_y) == 0:
        return HEIGHT // 2, WIDTH // 2

    # If multiple edge points, choose the one furthest from the center of the image
    center_pixel_y, center_pixel_x = HEIGHT // 2, WIDTH // 2
    distances = np.sqrt((edge_y - center_pixel_y)**2 + (edge_x - center_pixel_x)**2)
    
    furthest_idx = np.argmax(distances)
    
    return edge_y[furthest_idx], edge_x[furthest_idx]

def update(frame):
    global zoom, center_x, center_y, fractal_type, c_julia_x, c_julia_y # Add fractal_type and c_julia to global
    zoom *= gmpy2.mpfr('1.1')
    fractal_data = generate_fractal(center_x, center_y, zoom, WIDTH, HEIGHT, MAX_ITER, fractal_type, c_julia_x, c_julia_y)

    # Find new target every few frames and adjust center directly
    y_idx, x_idx = find_most_detailed_region(fractal_data)
    
    # Calculate the current scale per pixel
    current_scale = gmpy2.mpfr('1.5') / zoom
    scale_per_pixel_x = (gmpy2.mpfr('2') * current_scale) / gmpy2.mpfr(WIDTH)
    scale_per_pixel_y = (gmpy2.mpfr('2') * current_scale) / gmpy2.mpfr(HEIGHT)

    # Calculate the offset from the center of the image to the point of interest
    offset_x_pixels = gmpy2.mpfr(int(x_idx)) - gmpy2.mpfr(WIDTH) / gmpy2.mpfr('2')
    offset_y_pixels = gmpy2.mpfr(int(y_idx)) - gmpy2.mpfr(HEIGHT) / gmpy2.mpfr('2')

    offset_x_complex = offset_x_pixels * scale_per_pixel_x
    offset_y_complex = offset_y_pixels * scale_per_pixel_y

    # Smoothly adjust center_x and center_y towards the point of interest
    center_x += offset_x_complex * gmpy2.mpfr('0.1')
    center_y += offset_y_complex * gmpy2.mpfr('0.1')

    # Set data and color limits for binary (black and white) output
    image.set_data(fractal_data)
    image.set_clim(0, 1) # Set clim to 0-1 for binary output
    return [image]

def on_key_press(event):
    global zoom, center_x, center_y, fractal_type, c_julia_x, c_julia_y

    if event.key == 'left' or event.key == 'right':
        # Reset zoom
        zoom = gmpy2.mpfr('1.0')

        # Reselect a new random fractal type and parameters
        fractal_type = random.choice(['mandelbrot'] + ['julia'] * 9)
        if fractal_type == 'mandelbrot':
            center_x = gmpy2.mpfr('-0.743643887037151')
            center_y = gmpy2.mpfr('0.13182590420533')
            print("\nGenerating new Mandelbrot set.")
        else: # fractal_type == 'julia'
            center_x = gmpy2.mpfr('0.0')
            center_y = gmpy2.mpfr('0.0')
            c_julia_x, c_julia_y = random.choice(JULIA_CONSTANTS)
            print(f"\nGenerating new Julia set with c = {float(c_julia_x):.5f} + {float(c_julia_y):.5f}i")
        
        # Redraw the canvas to reflect the changes immediately
        fig.canvas.draw_idle()

ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=False)

# Connect the key press event handler
fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()
