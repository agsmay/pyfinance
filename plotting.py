import matplotlib.pyplot as plt

def startup_plotting(font_size=4, line_width=1, output_dpi=300, tex_backend=False, figure_size=(3, 2)):
    """
    Initializes the plot settings for nice-looking plots.

    Parameters:
    font_size (int): Default font size for the plots.
    line_width (float): Line width for the plots.
    output_dpi (int): Output DPI for saving figures.
    tex_backend (bool): If True, enables LaTeX rendering for text in plots.
    figure_size (tuple): Default figure size (width, height) in inches.
    """
    # Set LaTeX backend if requested
    if tex_backend:
        try:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "mathtext.default": "regular",
                "text.latex.preamble": [r'\usepackage{amsmath,amssymb,bm,lmodern}']
            })
        except:
            print("WARNING: LaTeX backend not configured properly. Not using LaTeX.")
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"]
            })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"]
        })
    
    # Set the default figure size (in inches)
    plt.rcParams["figure.figsize"] = figure_size
    
    # General plot styling settings
    plt.rcParams.update({
        "axes.grid": True,
        "legend.framealpha": 1,
        "legend.edgecolor": [1, 1, 1],
        "lines.linewidth": line_width,
        "savefig.dpi": output_dpi,
        "savefig.format": 'pdf',
        "figure.dpi": output_dpi
    })
    
    # Font size for different plot elements
    plt.rc('font', size=font_size)  # Controls default text size
    plt.rc('axes', titlesize=font_size)  # Font size of the title
    plt.rc('axes', labelsize=font_size)  # Font size of the x and y labels
    plt.rc('legend', fontsize=0.85 * font_size)  # Font size of the legend
    plt.rc('xtick', labelsize=0.85 * font_size)  # Font size of the x tick labels
    plt.rc('ytick', labelsize=0.85 * font_size)  # Font size of the y tick labels


# Call the startup_plotting function to apply settings by default
startup_plotting(tex_backend=True)
