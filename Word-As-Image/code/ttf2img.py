from PIL import Image, ImageDraw, ImageFont

def generate_character_image(character, font_path, output_path, image_size=(600, 600), text_color=(0, 0, 0), coverage_ratio=1):
    """
    Generate a 256x256 image with a single character using a specified TTF font.
    The character will be centered both horizontally and vertically, and scaled to cover approximately 70% of the image.

    Parameters:
        character (str): The single character to render.
        font_path (str): Path to the TTF font file.
        output_path (str): Path to save the generated image.
        image_size (tuple): Size of the image (width, height).
        text_color (tuple): Color of the text (R, G, B).
        coverage_ratio (float): Approximate ratio of the text size to the image size.

    Returns:
        None
    """
    if len(character) != 1:
        raise ValueError("Only one character can be rendered at a time.")
    
    try:
        # Create a blank image with a white background
        image = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(image)
        width, height = image_size

        # Dynamically calculate font size to achieve desired coverage ratio
        max_font_size = int(width * coverage_ratio)
        font_size = max_font_size
        font = ImageFont.truetype(font_path, font_size)

        # Adjust font size to fit the character within the desired coverage
        while True:
            text_width, text_height = draw.textsize(character, font=font)
            if text_width <= width * coverage_ratio and text_height <= height * coverage_ratio:
                break
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)

        # Calculate text position to center it
        # Use textsize and font metrics to precisely center the character
        ascent, descent = font.getmetrics()
        text_width, text_height = draw.textsize(character, font=font)
        text_offset = font.getoffset(character)
        vertical_correction = (text_height - (ascent - descent)) // 2  # Correct for vertical misalignment
        position = (
            (width - text_width) // 2 - text_offset[0],  # Adjust for horizontal offset
            (height - text_height) // 2 - text_offset[1] + vertical_correction  # Adjust for vertical offset
        )

        # Draw the character
        draw.text(position, character, fill=text_color, font=font)

        # Save the image
        image.save(output_path)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    # Example usage
    character_to_render = "今"
    ttf_path = "code/data/fonts/NotoSansJP-VariableFont_wght.ttf"  # Replace with your TTF font file path
    output_file = "../data/melody/11.png"  # Replace with your desired output file name
    generate_character_image(character_to_render, ttf_path, output_file)
