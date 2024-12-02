
# Face Recognition System with MTCNN and InceptionResnetV1

This project implements a real-time face recognition system using the **FaceNet PyTorch** library. It allows users to capture face images, train a recognition model, and perform live face recognition using a webcam.

## Features

- **Add a Person**: Capture facial images using the webcam and save them for training.
- **Train Model**: Train a face recognition model using MTCNN for face detection and InceptionResnetV1 for embeddings.
- **Live Recognition**: Perform real-time face recognition through a webcam and display the results.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MEDPRG/face-recognition-system.git
   cd face-recognition-system
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Script**: Start the program by running:
   ```bash
   python main.py
   ```

2. **Menu Options**:
   - **Option 1**: Add a Person  
     Enter a name, and the program will capture 90 images of the person using the webcam. The images are saved in the `photos/` directory.
   - **Option 2**: Train the Model  
     The program processes saved images, extracts face embeddings, and saves the data to `data.pt`.
   - **Option 3**: Live Camera Recognition  
     The system uses the webcam for real-time recognition, displaying names and confidence percentages.

---

## Folder Structure

```
.
├── photos/               # Directory for storing face images
├── data.pt               # Trained embeddings and names
├── main.py               # Main script for the program
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

---

## Dependencies

- `facenet-pytorch`
- `torch`
- `torchvision`
- `Pillow`
- `opencv-python`
- `uuid`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Notes

- Ensure the `photos/` directory exists before training (it will be created automatically if not present).
- The probability threshold for face recognition is set to **90%**.
- Pretrained models for MTCNN and InceptionResnetV1 will be downloaded automatically if not available.

---

## Future Enhancements

- Implement a graphical user interface (GUI) for ease of use.
- Add support for cloud storage to save and retrieve embeddings.
- Enhance recognition accuracy by fine-tuning thresholds.

---

## License

This project is licensed under the MIT License.

---

## Author

**MEDPRG**  
[GitHub Profile](https://github.com/MEDPRG)
