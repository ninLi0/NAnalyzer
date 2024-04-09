from PIL import Image
import pytesseract

# Path to the image file
image_path = 'data/lays.jpg'

# Open the image file
image = Image.open(image_path)

# Optional: preprocess the image for better OCR results
# image = preprocess_image(image)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Perform OCR on the image
extracted_text = pytesseract.image_to_string(image)

# Process the extracted text to find nutrition facts
# nutrition_facts = process_extracted_text(extracted_text)

# Print the extracted text
print(extracted_text)
