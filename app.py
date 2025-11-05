from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
import streamlit as st


def waste_classification(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model('keras_model.h5', compile=False)

    # Load the labels
    class_names = open('labels.txt', 'r').readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img.convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # print('Class:', class_name, end='')
    # print('Confidence score:', confidence_score)
    
    return class_name, confidence_score


st.set_page_config(layout = 'wide')

st.title("Waste Classifier App")

input_img = st.file_uploader("Enter your Image", type = ['jpeg', 'jpg', 'png'])

if input_img is not None:
    if st.button("Classify"):

        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.info("The image has been uploaded.")
            st.image(input_img, width = 400)

        with col2:
            st.info("Learn about where to throw your waste!")
            image_file = Image.open(input_img)
            label, confidence_score = waste_classification(image_file)

            if label == '0 Paper\n':
                st.write("This image is classified as Paper.")
                st.write("Proper disposal of paper helps conserve forests, reduce landfill waste, and lower greenhouse gas emissions. When paper is recycled instead of thrown away, it can be turned into new paper products, saving energy and water used in production. Mismanaged paper waste contributes to growing landfills and unnecessary deforestation.")
                st.write("Paper should be placed in recycling bins. This includes office paper, newspapers, notebooks, magazines, class worksheets, and any other paper products. Avoid disposing of paper with heavy contamination,  such as food residue, and ensure that all paper recycled is clean and dry. Otherwise, place the paper in landfill as it can't be recycled.")

            if label == '1 Plastic\n':
                st.write("This image is classified as Plastic.")
                st.write("Incorrect disposal of plastic leads to severe environmental damage, including ocean pollution and harm to wildlife. Recycling plastic reduces the need for new plastic production, conserving fossil fuels and lowering carbon emissions. Properly managing plastic waste is essential to limit its long-term persistence in ecosystems.")
                st.write("Plastic items like bottles, containers, and packaging should be placed in plastic recycling bins. Please clean and rinse plastic items as needed before disposal. Avoid placing non-recyclable plastics, such as plastic bags or certain wrappers, in recycling bins as they may require separate collection or go to landfill if unusable.")
            
            if label == '2 Glass\n':
                st.write("This image is classified as Glass.")
                st.write("Glass is infinitely recyclable, meaning it can be reused without loss of quality. Proper disposal prevents it from breaking in landfills, reducing hazards for sanitation workers and wildlife. Recycling glass also saves raw materials like sand, soda ash, and limestone, and reduces energy consumption.")
                st.write("Dispose of glass bottles and jars in glass recycling bins. Make sure they are empty and rinsed. Do not include mirrors, ceramics, or light bulbs, as these require specialized disposal methods.")

            if label == '3 Metal\n':
                st.write("This image is classified as Metal.")
                st.write("Metals like aluminum and steel can be recycled endlessly without losing quality. Proper disposal reduces mining demand, conserves natural resources, and prevents heavy metals from contaminating soil and water. Recycling metals also saves significant energy compared to producing new metal from ore.")
                st.write("Aluminum cans, tin cans, and clean metal scraps should go in metal recycling bins. Avoid mixing with non-metal materials. Large metal items may need to be taken to specialized recycling centers.")

            if label == '4 Organic\n':
                st.write("This image is classified as Organic Waste.")
                st.write("Organic waste such as food scraps and garden trimmings decomposes naturally and can produce nutrient-rich compost. Correct disposal reduces landfill methane emissions and supports soil health when composted. Mixing organics with other trash contaminates recycling streams and contributes to greenhouse gas emissions.")
                st.write("Organic waste should go in compost bins or organic waste collections. This includes fruit and vegetable peels, coffee grounds, and garden waste. Avoid plastics or non-compostable materials in these bins.")
            
            if label == '5 E-Waste\n':
                st.write("This image is classified as E-Waste.")
                st.write("Electronic waste contains valuable materials like gold and copper but also toxic substances like lead and mercury. Improper disposal can pollute soil and water and harm human health. Recycling e-waste recovers valuable metals and ensures hazardous components are handled safely.")
                st.write("E-waste, including phones, batteries, and old computers, should be taken to designated e-waste recycling centers or collection points. Do not throw electronics in regular trash, as they require specialized handling.")
            
            if label == '6 Textile/Clothing\n':
                st.write("This image is classified as Textile or Clothing.")
                st.write("Discarded textiles can take decades to decompose and often end up in landfills. Recycling or donating clothes reduces textile waste, conserves resources, and supports charitable initiatives. Proper sorting of clothing prevents contamination of recycling streams and encourages sustainable reuse.")
                st.write("Old or unwanted clothing should be placed in textile recycling bins, donation boxes, or second-hand stores. Ensure that the textile material is clean and dry before donation. Damaged fabrics may be repurposed as rags or sent to specialized textile recycling programs.")

            if label == '7 Landfill\n':
                st.write("This image is classified as Landfill.")
                st.write("Items that cannot be recycled or composted should go to landfill, but minimizing this waste is crucial to reduce environmental impact. Proper disposal prevents contamination of soil and water and ensures that hazardous materials are handled safely. Reducing landfill waste overall encourages a circular economy.")
                st.write("Landfill-bound items include heavily soiled packaging, certain plastics, and non-recyclable materials. These should be placed in regular trash bins for collection by municipal waste services. Avoid mixing recyclables with landfill waste whenever possible.")

            if label == '8 Cardboard\n':
                st.write("This image is classified as Cardboard.")
                st.write("Proper disposal of cardboard helps reduce landfill waste and allows the material to be recycled into new packaging products. Recycling cardboard saves trees, energy, and water, and prevents pollution that can result from decomposing cardboard in landfills. Flattening and cleaning cardboard before disposal makes recycling more efficient and effective.")
                st.write("Cardboard boxes and packaging should be placed in paper/cardboard recycling bins. Make sure to remove any plastic wrap, tape, or food residue. Heavily soiled cardboard, such as greasy pizza boxes, should instead go in organic or landfill bins, depending on the situation.")
        
        with col3:
            st.info("Arizona State University Waste Information")
            image_file = Image.open(input_img)
            label, confidence_score = waste_classification(image_file)

            if label == '0 Paper\n':
                st.write("Paper items can be recycled in any recycling blue bin on campus, including in offices, classrooms, and residence halls on Arizona State University Campus locations.")

            if label == '1 Plastic\n':
                st.write("Plastic items can be recycled in any recycling blue bin on campus, including in offices, classrooms, and residence halls on Arizona State University Campus locations.")

            if label == '2 Glass\n':
                st.write("Glass items (jars, bottles) can be recycled in any recycling blue bin on campus after empyting all liquid contents and rinsing properly. Locations include offices, classrooms, and residence halls on Arizona State University Campus locations.")
                    
            if label == '3 Metal\n':
                st.write("Metal items can be recycled in any recycling blue bin on campus, including in offices, classrooms, and residence halls on Arizona State University Campus locations.")
                
            if label == '4 Organic\n':
                st.write("Organic items, such as leftover food, can be placed in any compost container on campus. Locations include: ")
                st.write("1. Memorial Union - Food Court and Plaza : https://www.google.com/maps?&daddr=33.41777,-111.93438")
                st.write ("2. Hassayampa Dinning Hall: https://www.google.com/maps?&daddr=33.41626,-111.92868")
                st.write("3. Sun Devil Athletic Stadiums")
                st.write("4. Residential halls")
                st.write("5. Community kitchens and dinning halls")
                
            if label == '5 E-Waste\n':
                st.write("E-Waste items, including old computers and electronics, can be picked up at one of the following ASU locations after submitting a request:  ")
                st.write("1. Memorial Union: https://www.google.com/maps?&daddr=33.41777,-111.93438 ")
                st.write("2. Fletcher Library: https://www.google.com/maps?&daddr=33.60739,-112.15988")
                st.write("3. Engineering Center ET Drop Off: https://www.google.com/maps?&daddr=33.41882,-111.93242")
                st.write("4. Health Futures Center ET Drop Off: https://www.google.com/maps/dir//33.657091,-111.949388/@33.6570637,-112.0317892,12z?entry=ttu&g_ep=EgoyMDI1MTAyOS4yIKXMDSoASAFQAw%3D%3D")
                st.write("5. Quad 1 ET Drop Off: https://www.google.com/maps?&daddr=33.308275,-111.680445")
                st.write("6. University Center ET Drop Off: https://www.google.com/maps?&daddr=33.452858,-112.072936")
                st.write("7. University Services Building ET Drop Off: https://www.google.com/maps?&daddr=33.41156,-111.925606")
                
            if label == '6 Textile/Clothing\n':
                st.write("Old clothing or textile material can be donated at any donation site.")
                st.write("ASU students can drop off donation items, including cloths, shoes, accessories, linens, and more, at the Big Brothers Big Sisters collection boxes.")
                st.write("On campus ASU locations of donation boxes can be found here: https://cfo.asu.edu/ditch-the-dumpster")

            if label == '7 Landfill\n':
                st.write("Items that can not be recycled, composted, reused, or donated can be discarded in any landfill container on ASU campus.")
                st.write("Please consider the items that go into the landfill and consider using more environmentally-friendly materials in the future!")
            
            if label == '8 Cardboard\n':
                st.write("Smaller quantities of cardboard can be recycled in any recycling blue bins nearest to you. Please flatten out boxes before recycling.")
                st.write("For cardboard sections that are greasy or contaminated with food residue, such as pizza boxes, please remove unusable sections and place in landfill.")
                st.write("Larger quantities of cardboard can be picked up from ASU campus. Please submit a request here: https://webtma-support.asu.edu/FDMServiceRequest/Default.aspx")