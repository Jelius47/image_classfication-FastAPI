# Digit Image Classifier with FastAPI

This project is a **REST API** built using **FastAPI** that classifies handwritten digits (0-9) from images. It uses a pre-trained machine learning model (loaded from a `.pkl` file) to predict the digit in an uploaded image. The API is designed to be simple, fast, and easy to integrate with frontend applications.

---

## Features

- **Image Upload**: Accepts image files (e.g., PNG, JPEG) for digit classification.
- **Pre-trained Model**: Uses a pre-trained model (`mnist_model_image_classifier.pkl`) for predictions.
- **CORS Support**: Allows cross-origin requests for integration with frontend applications.
- **Fast and Lightweight**: Built with FastAPI for high performance and scalability.

---

## Technologies Used

- **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python.
- **Pillow (PIL)**: For image processing (conversion, resizing, and inversion).
- **NumPy**: For converting images into arrays compatible with the machine learning model.
- **Scikit-learn**: The machine learning framework used to train the model.
- **Pickle**: For loading the pre-trained model from a `.pkl` file.

---

## Setup Instructions

### Prerequisites

1. **Python 3.7+**: Ensure Python is installed on your system.
2. **Dependencies**: Install the required Python packages.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/digit-classifier-api.git
   cd digit-classifier-api
   
2. Navigate to index.html and open it to your browser then your good to go with starting your server

#### Install the dependencies:
```bash
pip install -r requirements.txt
```
#### Run the fastApi-server
```bash
uvicorn main:app --reload
```
### API Endpoints
1. Predict Digit

    Endpoint: POST /predict_image/

    Description: Accepts an image file and returns the predicted digit.

    Request:

        Body: file (image file in PNG, JPEG, etc.)

    Response:
 ```json
    {
        "prediction": 5
    }

```
### Project structure

```bash
digit-classifier-api/
├── main.py #FastAPI application code
├──images/                #Folder with images
├── mnist_model_image_classifier.pkl  # Pre-trained model
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```
### Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

    Fork the repository.

    Create a new branch (git checkout -b feature/YourFeature).

    Commit your changes (git commit -m 'Add some feature').

    Push to the branch (git push origin feature/YourFeature).

    Open a pull request.

### Acknowledgments

    FastAPI for providing a robust and easy-to-use framework.

    Scikit-learn for the machine learning tools used to train the model.

    Pillow for image processing capabilities.


### Contacts

For questions or feedback, feel free to reach out:

    Name: Jelius Heneriko

    Email: jeliusheneriko47@gmail.com

Happy coding! 🚀