## Imports all the needed libraries for the program
## Note the system will need to have installed keras, tensorflow, image, and numpy
from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model 
from keras.applications.vgg16 import VGG16 
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def create_img():
    ## Sets the specifications of the data generator 
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.4,
            horizontal_flip=True,
            fill_mode='nearest')

    import os 
    ##Current Working Directory
    print('Example directory:', os.getcwd())  
    inp_dir = input ("Enter the folder location of the pictures you want to create: ")
    os.chdir(inp_dir) ##Change with your current working directory
    
    #For each picture in the directory 
    for path in os.listdir():
        img = load_img(f"{path}")
        x = img_to_array(img)    # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=".", save_prefix='img', save_format='jpeg'):
            i += 1
            if i > 10:     ## creates 10 image form 1 image 
                break  

def create_mdl():
    ### Defining Image size
    IMAGE_SIZE = [224, 224]

    ### Loading model
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    ### Freezing layers
    for layer in vgg.layers:  
      layer.trainable = False

    ### adding a 3 node final layer for predicion
    x = Flatten()(vgg.output)
    prediction = Dense(3, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)

    ### Generating Summary
    model.summary()

    ## Compile the model using categorical crossentropy and the adam optimizer
    model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ## Creating the datagens for traning 
    train_dir = input("Enter the directory of the three classifier folders: ")
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    training_set = train_datagen.flow_from_directory(train_dir, target_size = (224, 224), batch_size = 32, class_mode = 'categorical', color_mode="rgb")

    ## Creating the datagens for testing 
    test_dir = input("Enter the directory of the three testing folders: ")
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(test_dir, target_size = (224, 224), batch_size=16, class_mode = 'categorical')

    ## Uses the fit method to generate the model
    ## Note this will take a long time to fit the model, and will depend on your system specs
    ## Changing the epochs will shorten compile time, but lessen the accuracy 
    r = model.fit(training_set, validation_data=test_set,  epochs=25, steps_per_epoch=len(training_set), validation_steps=len(test_set))

    ## Saves the model 
    model.save("ripeness.h5")

# Main function where the user interacts
def main():
    ans = ''
    while ans != "ex":
        print("-------------------------------------------------------------------------------------------------------------------\n")
        ans = input("Would you like to create images or create model [ci/cm/ex]: ")

        print("-------------------------------------------------------------------------------------------------------------------\n")
        if ans == "ci":
            create_img()
            print("Images Created")
        elif ans == "cm":
            create_mdl()
            print("Model Created")
        elif ans == 'ex':
            break
        else:
            print('That is not a valid answer. Please answer [ci/cm/ex]!')

main()