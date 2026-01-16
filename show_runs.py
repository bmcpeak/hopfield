import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import numpy as np


def show_all_runs(folder='runs20'):
    """Display all pngs in runs folder in a grid."""
    files = sorted(glob.glob(f'{folder}/*.png'))

    if not files:
        print(f"No pngs found in {folder}/")
        return

    n = len(files)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = np.array(axes).flatten()  # force it to be a flat array

    for i, f in enumerate(files):
        img = mpimg.imread(f)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(os.path.basename(f), fontsize=10)

    # Hide extra axes if odd number of files
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show_all_runs('runs20')
 #   show_all_runs('runs30')