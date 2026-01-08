# iSCAT Video Simulation

This script generates realistic videos and corresponding segmentation masks of particles undergoing free 3D Brownian motion as observed by an interferometric scattering (iSCAT) microscope. The simulation is highly configurable, with Python parameters available to control the optical setup, particle properties, noise levels, and output video characteristics.

## Features

-   **Physics-Based Simulation**: Accurately models Brownian motion using the Stokes-Einstein equation.
-   **Vectorial iPSF**: Computes the interferometric Point Spread Function using Mie scattering theory and a Debye-Born integral model.
-   **Customizable Parameters**: Easily modify particle size, refractive index, optical setup (NA, wavelength), and system aberrations.
-   **Realistic Noise**: Simulates both Poisson (shot) noise and Gaussian (read) noise to mimic real camera performance.
-   **Motion Blur**: Includes multi-sample motion blur to simulate particle movement during a single frame's exposure time.
-   **Mask Generation**: Automatically generates corresponding binary segmentation masks for each particle in every frame, ideal for training machine learning models.

## Setup & Installation

To run this script, it is highly recommended to create a dedicated Python environment to avoid dependency conflicts. The following instructions use Conda.

1.  **Create a new Conda environment:**
    *Note: The environment was developed with Python 3.12.*
    ```bash
    conda create -n sim_env python=3.12
    ```

2.  **Activate the environment:**
    ```bash
    conda activate sim_env
    ```

3.  **Install Required Packages:**
    For guaranteed compatibility, you should install the exact versions of the packages used during development.

    **Method A: Direct Installation**

    Run the following command to install the specific versions:
    ```bash
    pip install numpy==1.26.4 matplotlib==3.10.0 opencv-python==4.10.0 scipy==1.16.0 tqdm==4.67.1
    ```

    **Method B: Using `requirements.txt` (Recommended)**

    Create a file named `requirements.txt` in the same directory as your script and paste the following content into it:
    ```
    numpy==1.26.4
    matplotlib==3.10.0
    opencv-python==4.10.0
    scipy==1.16.0
    tqdm==4.67.1
    ```
    Then, install all packages from this file with a single command:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Configure the Simulation:**
    Open the `iscat.py` script and modify the values in the `PARAMS` dictionary at the top of the file. Here you can set the image size, particle count, noise levels, video duration, and much more.

2.  **Execute the Script:**
    Run the script from your terminal.
    ```bash
    python iscat.py
    ```
    The simulation will print its progress to the console.

## Output

The script will generate the following files and directories on your Desktop by default (this can be changed in the `PARAMS` dictionary):

-   **`iscat_simulation.mp4`**: The final output video file showing the background-subtracted iSCAT signal.
-   **`/iscat_masks/`**: A directory containing sub-folders for each particle (e.g., `particle_1`, `particle_2`). Inside each sub-folder are the individual `.png` mask files for every frame of the video.
