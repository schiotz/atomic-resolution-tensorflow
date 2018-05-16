import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def discrete_cmap(N, base_cmap='Paired'):
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    #return base.from_list(cmap_name, color_list, N)
    return ListedColormap(color_list, cmap_name, N)
