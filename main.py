import io
import pickle
import numpy as np
import PIL.ImageOps
import PIL.Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model from the pickle file
with open("mnist_model_image_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to predict the digit in an uploaded image.

    Args:
        file (UploadFile): The image file uploaded by the user.

    Returns:
        dict: A dictionary containing the predicted digit.
    """
    # Read the uploaded file
    contents = await file.read()

    # Convert the image to grayscale using PIL
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert("L")

    # Invert the image colors (scikit-learn and PIL read images differently)
    pil_image = PIL.ImageOps.invert(pil_image)

    # Resize the image to 28x28 pixels (MNIST dataset size)
    pil_image = pil_image.resize((28, 28), PIL.Image.ANTIALIAS)

    # Convert the image to a NumPy array and reshape it for the model
    image_array = np.array(pil_image).reshape(1, -1)

    # Predict the digit using the loaded model
    prediction = model.predict(image_array)

    # Return the prediction as a JSON response
    return {"prediction": int(prediction[0])}