from PIL import Image
import pytesseract
import re
from pyNutriScore import NutriScore

# Path to the image file
# image_path = 'data/lays.jpg'
image_path = 'data/b2.png'
# image_path = 'data/b3.jpg'
# image_path = 'data/b1.png'


# Optional: preprocess the image for better OCR results
def preprocess_image(image_path):
    from PIL import Image
    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(image_path)
    # image = oldImage

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image using Otsu's method
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew the image if needed
    # (You would need to implement/find a function to calculate the skew angle and rotate the image)

    # Apply dilation or erosion if needed
    # (Choose appropriate kernel size and operation based on the text appearance)
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary, kernel, iterations=1)  # or cv2.erode()
    return processed_image
    # Save the processed image
    # cv2.imwrite('processed_image.png', processed_image)

image = preprocess_image(image_path)



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Perform OCR on the image
extracted_text = pytesseract.image_to_string(image)
print(extracted_text)
def parse_nutrition_facts(text):
    # Split the text into lines and initialize an empty dictionary
    lines = text.split('\n')
    nutrition_facts = {
        'serving_size': 0.0,
        'calories': 0.0,
        'energy': 0.0,
        'total_fat': 0.0,
        'saturated_fats': 0.0,
        'total_carbohydrate': 0.0,
        'sugar': 0.0,
        'sodium': 0.0,
        'fibers': 0.0,
        'proteins': 0.0,
        'fruit_percentage': 0.0,
        'cholesterol': 0.0,
        'vitamin_d': 0.0,
        'calcium': 0.0,
        'iron': 0.0,
        'potassium': 0.0,
        'Vitamin C': 0.0,
        'servings_per_container': 0.0,
        'trans_fat': 0.0,
    }

    # Helper function to extract numeric values
    def extract_value(line):
        # Find all number-like substrings (integers or floats)
        numbers = re.findall(r'\d+\.?\d*', line)
        # Return the first found number as a float, or 0.0 if  are found
        return float(numbers[0]) if numbers else 0.0
    def extract_valueSp(line):
        # Find all number-like substrings (integers or floats)
        numbers = re.findall(r'\d+\.?\d*', line)
        # Return the first found number as a float, or 0.0 if  are found
        return float(numbers[-1]) if numbers else 0.0

    # correction factor to match the library
    correction = 0.0
    # Iterate over the lines and fill the dictionary with data
    for i in range(len(lines)):
        line = lines[i].lower()  # Convert line to lowercase for easier matching
        if 'servings per container' in line:
            nutrition_facts['servings_per_container'] = (extract_value(line))
        elif 'serving size' in line:
            nutrition_facts['serving_size'] = line.split('size')[1].strip()
            # library computes per 100g, thus correction factor is needed
            correction = 100/extract_valueSp(line)
            # correction = 1
        elif 'calories' in line and nutrition_facts['calories']==0:
            nutrition_facts['calories'] = (extract_value(line))*correction
            nutrition_facts['energy'] = nutrition_facts['calories']*4.184
        elif 'total fat' in line and nutrition_facts['total_fat']==0:
            nutrition_facts['total_fat'] = (extract_value(line))*correction
        elif 'saturated fat' in line and nutrition_facts['saturated_fats']==0:
            nutrition_facts['saturated_fats'] = (extract_value(line))*correction
        elif 'trans fat' in line:
            nutrition_facts['trans_fat'] = (extract_value(line))*correction
        elif 'cholesterol' in line:
            nutrition_facts['cholesterol'] = (extract_value(line))*correction
        elif 'sodium' in line and nutrition_facts['sodium']==0:
            nutrition_facts['sodium'] = (extract_value(line))*correction
        elif 'total carbohydrate' in line and nutrition_facts['total_carbohydrate']==0:
            nutrition_facts['total_carbohydrate'] = (extract_value(line))*correction
        elif 'dietary fiber' in line:
            nutrition_facts['fibers'] = (extract_value(line))*correction
        elif 'total sugars' in line and nutrition_facts['sugar']==0:
            nutrition_facts['sugar'] = (extract_value(line))*correction
        elif 'protein' in line:
            nutrition_facts['proteins'] = (extract_value(line))*correction
        elif 'vitamin d' in line:
            nutrition_facts['vitamin_d'] = (extract_value(line))*correction
        elif 'calcium' in line:
            nutrition_facts['calcium'] = (extract_value(line))*correction
        elif 'iron' in line or 'lron' in line:
            nutrition_facts['iron'] = (extract_value(line))*correction
        elif 'potassium' in line and nutrition_facts['potassium']==0:
            nutrition_facts['potassium'] = (extract_value(line))*correction
    return nutrition_facts


# printing the scanned nutrition facts
nutrition_facts = parse_nutrition_facts(extracted_text)
for k,v in nutrition_facts.items():
    print(k, v)
print('\n')

# calculating overall rating
result = NutriScore().calculate_class(nutrition_facts, 'solid')
print("NutriScore result:",result)

# testing code
lol = { #*2.04
        'serving_size': 49,
        'calories': 210*2.04,
        'energy': 210*4.184*2.04,
        'total_fat': 6*2.04,
        'saturated_fats': 1.5*2.04,
        'total_carbohydrate': 37.0*2.04,
        'sugar': 4*2.04,
        'sodium': 430*2.04,
        'fibers': 2.0*2.04,
        'proteins': 4*2.04,
        'fruit_percentage': 0.0,
        'cholesterol': 0.0,
        'vitamin_d': 0.0,
        'calcium': 0.0,
        'iron': 0.0,
        'potassium': 0.0,
        'Vitamin C': 0.0,
        'servings_per_container': 0.0,
        'trans_fat': 0.0,
    }


'''
algorithm basically works now?
what should be next up? 
'''
# print(NutriScore().calculate_class(lol, 'solid'))
# Process the extracted text to find nutrition facts
# nutrition_facts = process_extracted_text(extracted_text)

# Print the extracted text