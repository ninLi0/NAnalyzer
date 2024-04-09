from PIL import Image
import pytesseract
import re
from pyNutriScore import NutriScore

# Path to the image file
image_path = 'data/lays.jpg'
# image_path = 'data/b2.png'

# Open the image file
# image = Image.open(image_path)

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
# print(extracted_text)
def parse_nutrition_facts(text):
    # Split the text into lines and initialize an empty dictionary
    lines = text.split('\n')
    nutrition_facts = {
        'servings_per_container': None,
        'serving_size': None,
        'calories': None,
        'energy': None,
        'total_fat': None,
        'saturated_fat': None,
        'trans_fat': None,
        'cholesterol': None,
        'sodium': None,
        'total_carbohydrate': None,
        'dietary_fiber': None,
        'sugar': None,
        'protein': None,
        'vitamin_d': None,
        'calcium': None,
        'iron': None,
        'potassium': None,
        'Vitamin C': None,
    }

    # Helper function to extract numeric values
    # def extract_value(line):
    #     number_str = ''.join(filter(str.isdigit, line.split()[0]))
    #     return float(number_str) if number_str else 0.0
    #     # return float(''.join(filter(str.isdigit, line.split()[0])))

    def extract_value(line):
        # Find all number-like substrings (integers or floats)
        numbers = re.findall(r'\d+\.?\d*', line)
        # Return the first found number as a float, or 0.0 if none are found
        return float(numbers[0]) if numbers else 0.0

    # Iterate over the lines and fill the dictionary with data

    for i in range(len(lines)-3):
        line = lines[i].lower()  # Convert line to lowercase for easier matching
        if 'servings per container' in line:
            nutrition_facts['servings_per_container'] = int(extract_value(line))
        elif 'serving size' in line:
            nutrition_facts['serving_size'] = line.split('size')[1].strip()
        elif 'calories' in line:
            nutrition_facts['calories'] = int(extract_value(line))
            nutrition_facts['energy'] = ((extract_value(line))/238.85)
        elif 'total fat' in line:
            nutrition_facts['total_fat'] = {'amount': extract_value(line), 'unit': 'g'}
        elif 'saturated fat' in line:
            nutrition_facts['saturated_fat'] = {'amount': extract_value(line), 'unit': 'g'}
        elif 'trans fat' in line:
            nutrition_facts['trans_fat'] = {'amount': extract_value(line), 'unit': 'g'}
        elif 'cholesterol' in line:
            nutrition_facts['cholesterol'] = {'amount': extract_value(line), 'unit': 'mg'}
        elif 'sodium' in line:
            nutrition_facts['sodium'] = {'amount': extract_value(line), 'unit': 'mg'}
        elif 'total carbohydrate' in line:
            nutrition_facts['total_carbohydrate'] = {'amount': extract_value(line), 'unit': 'g'}
        elif 'dietary fiber' in line:
            nutrition_facts['dietary_fiber'] = {'amount': extract_value(line), 'unit': 'g'}
        elif 'total sugars' in line:
            nutrition_facts['sugar'] = {'amount': extract_value(line), 'unit': 'g'}
        elif 'protein' in line:
            nutrition_facts['protein'] = {'amount': extract_value(line), 'unit': 'g'}
        elif 'vitamin d' in line:
            nutrition_facts['vitamin_d'] = {'amount': extract_value(line), 'unit': 'mcg'}
        elif 'calcium' in line:
            nutrition_facts['calcium'] = {'amount': extract_value(line), 'unit': 'mg'}
        elif 'iron' in line or 'lron' in line:
            nutrition_facts['iron'] = {'amount': extract_value(line), 'unit': 'mg'}
        elif 'potassium' in line:
            nutrition_facts['potassium'] = {'amount': extract_value(line), 'unit': 'mg'}
    return nutrition_facts


nutrition_facts = parse_nutrition_facts(extracted_text)
for k,v in nutrition_facts.items():
    print(k, v)
print('\n')

result = NutriScore().calculate_class(nutrition_facts, 'solid')
print(result)
# Process the extracted text to find nutrition facts
# nutrition_facts = process_extracted_text(extracted_text)

# Print the extracted text