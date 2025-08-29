
import numpy as np

def gaussian_kernel(sigma):
    """
    Zero-mean, L2-normalized 2D Gaussian (possibly elongated & rotated).
    sx, sy: std dev in x/y (pixels).
    """
    sx = sy = sigma
    
    radius = sigma * 2.8
    # Kernel size: 6 times the std dev, ensuring it's odd
    kx = int(radius * 2 + 1)
    ky = int(radius * 2 + 1)

    # Create coordinate grids
    x = np.linspace(-kx // 2, kx // 2, kx)
    y = np.linspace(-ky // 2, ky // 2, ky)
    X, Y = np.meshgrid(x, y)
    
    # Compute the Gaussian function
    G = np.exp(-(X**2 / (2 * sx**2) + Y**2 / (2 * sy**2)))
    
    # Normalize to ensure the sum is 1
    G /= np.max(G)
    G -= 0.55  # zero-mean adjustment

    dist = np.sqrt(X **2 + Y **2)
    circle_mask = dist <= radius
    G *= circle_mask

    return G
