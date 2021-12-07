# load_model_sample.py
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np




# load model
model = load_model("model")

#img size
img_width=324
img_height=324



#input generator
inputdataGen = ImageDataGenerator(rescale=1. / 255)

input = inputdataGen.flow_from_directory("img",target_size=(img_width, img_height),batch_size=1, class_mode=None)



# check prediction
    
pred = model.predict(input)

print(pred)

arr=pred
number=0
leafs=[
        "Ulmus carpinifolia","Sorbus aucuparia"
        ,"Salix sinerea","Populus","Tilia","Sorbus intermedia", "Fagus silvatica","Acer","Salix aurita","Quercus","Alnus incana"
        ,"Betula pubescens"," Salix alba","Populus tremula","Ulmus glabra"
    ]
# leafs=[
#         "Ulmus carpinifolia","Populus tremula","Ulmus glabra","Sorbus aucuparia","Alnus incana"
#         ,"Betula pubescens"," Salix alba","Acer","Salix aurita","Quercus"
#         ,"Salix sinerea","Populus","Tilia","Sorbus intermedia", "Fagus silvatica"
#     ]

for leaf in arr:
    number+=1
    
    maxElement = np.amax(leaf)
    certain=str(maxElement.item()*100)+" %"
    index = np.where(leaf == maxElement)[0]
    indextext= index[0].item()
    print(indextext)
    
    


    print("Number: "+ str(number) + ", Certain: "+ certain+ ", Guess: "+ leafs[indextext])


#l14 -> index 5 ofwel leaf 6
#l10 -> index 1
#l14 -> index 2
#l8 -> index 13




