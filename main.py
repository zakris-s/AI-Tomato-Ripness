import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

## This is the path of the image you want to test
## path =(r"F:\School\PNW\SEM7 SPR2022\ITS490\Tomtato Pics\Tomtato Pics\Test\Screen Shot 2022-02-11 at 2.56.16 AM.png")


## Function to import the data model created from the first program
def predict_stage(image_data,model):

    ## Sets the size of the image 
    size = (224, 224)

    ## Smooths out the image corners
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)

    ##Imports the image into a numpy arrayy and normailzes
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    ## Sets the ourput values equal to ripe, unripe, or overripe
    preds = ""
    prediction = model.predict(data)
    print (np.argmax(prediction))
    if np.argmax(prediction)==0:
        print("Unripe")
    elif np.argmax(prediction)==1:
        print("Overripe")
    else :
        print("Ripe")

    ## Returns the Prediction of the model
    return prediction

## Creates the main funtction for the program
def main():
    ## Uses the image library to open the above path 
    path = input("Enter the location of the image you want to test: ")
    image = Image.open(path)

    ## Loads the model created from the last program 
    model = tf.keras.models.load_model('ripeness.h5')

    ## Prints the prediction of the model using the above function
    prediction = predict_stage(image, model)
    ##print("Scale : (0: Unripe, 1: Overripe, 2: Ripe")
    print(prediction)

main()

