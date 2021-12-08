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

l=[
    "l1","l10","l11","l12","l13","l14","l15","l2","l3","l4","l5","l6","l7","l8","l9"
]


for leaf in arr:
    number+=1
    
    maxElement = np.amax(leaf)
    certain=str(maxElement.item()*100)+" %"
    index = np.where(leaf == maxElement)[0]
    indextext= index[0].item()

    print("Activated node index: "+str(indextext))
    
    


    print(str(number)+" : LeafNumber: "+ l[indextext] + ", Certain: "+ certain+ ", Guess: "+ leafs[indextext])





