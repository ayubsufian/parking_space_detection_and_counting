# Parking Space Detection and Counting

This project is a complete, end-to-end solution for real-time parking space detection and monitoring. It uses a deep learning model (VGG16) to classify individual parking spots as empty or occupied and serves the results through a dynamic web interface.
The application can process video from multiple sources, including local files, webcams, and IP cameras via RTSP streams.

## Features

- **Deep Learning Model**: Utilizes a VGG16-based Convolutional Neural Network (CNN) for high-accuracy classification of parking spots.
- **Web-Based Interface**: A clean, modern UI built with Flask and Bootstrap to display the live video feed and occupancy statistics.
- **Real-Time Dashboard**: Shows a live count of available and occupied parking spaces.
- **Multi-Source Video Support**: Seamlessly switch between a pre-recorded video, a live webcam, or an RTSP stream from an IP camera.
- **Complete ML Pipeline**: Includes scripts for training, evaluation, and defining parking spot locations.

## How It Works

The system operates on a complete machine learning pipeline:

1. **Model Training (`train/train.py`)**: A VGG16 model is trained using transfer learning on a dataset of images categorized into 'empty' and 'occupied' folders. Data augmentation is used to improve robustness. The final model is saved as `model/model_final.h5`.
2. **Parking Space Definition (`main/dataCollection.py`)**: A utility script provides a GUI to draw bounding boxes over a static image of the parking lot. The coordinates of these boxes are saved in `model/carposition.pkl`.
3. **Real-Time Detection & Deployment (`main.py`)**:
   * A Flask web server (using `waitress` for production) is launched.
   * It reads a video stream and the saved parking spot coordinates.
   * For each frame, it crops every parking spot, preprocesses it, and feeds it to the trained Keras model for prediction.
   * The processed video with color-coded bounding boxes (green for empty, red for occupied) is streamed to the web interface.
   * An API endpoint provides JSON data for the live space count, which is updated on the frontend using JavaScript.

## Setup and Installation

Follow these steps to set up the project locally.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayubsufian/parking_space_detection_and_counting.git
   cd parking_space_detection_and_counting
   ```
   
2. **Create and activate a virtual environment:**
   ```bash
   # Create the environment
   python -m venv venv

   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage Workflow

The project can be run in several stages. A pre-trained model and position file are included, so you can skip directly to Step 3.

### Step 1: Train a New Model (Optional)

If you have your own dataset of empty and occupied parking spot images, place them in the data/train and data/test directories. Then, run the training script:
```bash
python train/train.py
```
This will generate a new model/model_final.h5 file.

### Step 2: Define Parking Spaces

To define parking spaces for your specific parking lot video/image, run the data collection script:
```bash
python main/dataCollection.py
```
A window will open with the image assets/car1.png.
Left-click to add a new parking spot rectangle.
Right-click on an existing spot to delete it.
The positions are automatically saved to model/carposition.pkl. Close the window when finished.

### Step 3: Run the Web Application
This is the main step to launch the live monitoring system.
```bash
python main.py
```
The application will be served by waitress.
Open your web browser and navigate to http://127.0.0.1:8000.
You can use the UI to switch between the default video, your webcam, or provide a custom RTSP stream URL.

## Technologies Used

- **Backend**: `Flask`, `Waitress`
- **Deep Learning**: `TensorFlow`, `Keras`
- **Computer Vision**: `OpenCV`
- **Numerical Computing**: `NumPy`
- **Frontend**: `HTML`, `Bootstrap`, `jQuery`
