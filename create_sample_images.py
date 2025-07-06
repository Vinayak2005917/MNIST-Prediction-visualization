"""
Script to create some sample digit images for testing the MNIST classifier
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_digit_image(digit, filename, size=(100, 100)):
    """Create a simple image with a digit"""
    # Create a white image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        # Try to use a larger font size
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Calculate text position to center it
    if font:
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = 20, 20  # rough estimate for default font
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw the digit in black
    draw.text((x, y), str(digit), fill='black', font=font)
    
    return img

def main():
    # Create sample_images directory if it doesn't exist
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Create sample images for digits 0-9
    for digit in range(10):
        img = create_sample_digit_image(digit, f"sample_{digit}.png")
        img.save(os.path.join(sample_dir, f"sample_{digit}.png"))
        print(f"Created sample image for digit {digit}")
    
    # Create a few variations
    variations = [
        (3, "sample_3_alt.png"),
        (7, "sample_7_alt.png"),
        (8, "sample_8_alt.png"),
    ]
    
    for digit, filename in variations:
        img = create_sample_digit_image(digit, filename, size=(80, 80))
        img.save(os.path.join(sample_dir, filename))
        print(f"Created alternative sample image: {filename}")
    
    print(f"\nCreated {10 + len(variations)} sample images in the '{sample_dir}' folder")
    print("You can now use these images in the Local Library tab of your Streamlit app!")

if __name__ == "__main__":
    main()
