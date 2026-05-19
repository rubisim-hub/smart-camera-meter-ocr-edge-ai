"""
REALISTIC METER FACE SYNTHETIC DATA GENERATOR V3
==============================================
Generates photorealistic electricity meter images that match real-world characteristics:
- Multiple font styles (7-segment LCD, mechanical counter, digital display)
- Realistic backgrounds (bezel, frame, surrounding text)
- Lighting variations (shadows, reflections, highlights)
- Physical effects (perspective, tilt, blur, noise)
- Branding and labels (kWh, manufacturer text, property labels)

Output: 100,000 images (800x200) + validation.csv + preview grid
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import csv
from tqdm import tqdm

# ==================== CONFIGURATION ====================
OUTPUT_DIR = "realistic_meter_dataset"
NUM_SAMPLES = 100000
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 200
PREVIEW_SAMPLES = 20

# Character set (digits only for meter readings)
CHARS = "0123456789.-"

# ==================== METER PATTERNS ====================
METER_PATTERNS = [
    {"type": "7digit", "pattern": lambda: f"{random.randint(0, 9999999):07d}", "weight": 35},
    {"type": "decimal", "pattern": lambda: f"{random.randint(0, 999999):06d}.{random.randint(0, 9)}", "weight": 25},
    {"type": "hyphen", "pattern": lambda: f"{random.randint(0, 9999):04d}-{random.randint(0, 999):03d}", "weight": 15},
    {"type": "8digit", "pattern": lambda: f"{random.randint(0, 99999999):08d}", "weight": 15},
    {"type": "5digit", "pattern": lambda: f"{random.randint(0, 99999):05d}", "weight": 10},
]

# ==================== VISUAL STYLES ====================
VISUAL_STYLES = {
    "lcd_7segment": {"weight": 40, "bg_color": (20, 20, 20), "fg_color": (255, 50, 50)},
    "lcd_blue": {"weight": 25, "bg_color": (10, 30, 60), "fg_color": (100, 200, 255)},
    "lcd_green": {"weight": 20, "bg_color": (15, 25, 15), "fg_color": (50, 255, 100)},
    "mechanical": {"weight": 10, "bg_color": (240, 235, 220), "fg_color": (20, 20, 20)},
    "white_on_black": {"weight": 5, "bg_color": (0, 0, 0), "fg_color": (255, 255, 255)},
}

# ==================== 7-SEGMENT DISPLAY CLASS ====================
class SevenSegmentDisplay:
    """Renders 7-segment LCD digits with configurable styling"""
    
    def __init__(self, digit_width=40, digit_height=80, segment_thickness=8, spacing=10):
        self.digit_width = digit_width
        self.digit_height = digit_height
        self.thickness = segment_thickness
        self.spacing = spacing
        
        # Segment definitions (7-segment + decimal point)
        self.segments = {
            'a': [(0.15, 0.05), (0.85, 0.05), (0.75, 0.15), (0.25, 0.15)],  # top
            'b': [(0.85, 0.05), (0.95, 0.15), (0.85, 0.45), (0.75, 0.50)],  # top-right
            'c': [(0.85, 0.50), (0.95, 0.55), (0.95, 0.85), (0.75, 0.95)],  # bottom-right
            'd': [(0.15, 0.95), (0.85, 0.95), (0.75, 0.85), (0.25, 0.85)],  # bottom
            'e': [(0.05, 0.55), (0.15, 0.50), (0.25, 0.85), (0.15, 0.95)],  # bottom-left
            'f': [(0.05, 0.15), (0.15, 0.05), (0.25, 0.15), (0.15, 0.45)],  # top-left
            'g': [(0.25, 0.50), (0.75, 0.50), (0.65, 0.55), (0.35, 0.55)],  # middle
        }
        
        # Digit to segment mapping
        self.digit_segments = {
            '0': ['a', 'b', 'c', 'd', 'e', 'f'],
            '1': ['b', 'c'],
            '2': ['a', 'b', 'd', 'e', 'g'],
            '3': ['a', 'b', 'c', 'd', 'g'],
            '4': ['b', 'c', 'f', 'g'],
            '5': ['a', 'c', 'd', 'f', 'g'],
            '6': ['a', 'c', 'd', 'e', 'f', 'g'],
            '7': ['a', 'b', 'c'],
            '8': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            '9': ['a', 'b', 'c', 'd', 'f', 'g'],
            '-': ['g'],
            '.': [],  # Decimal point handled separately
        }
    
    def draw_digit(self, draw, x, y, char, fg_color, bg_color, dim_inactive=True):
        """Draw a single 7-segment digit"""
        if char not in self.digit_segments:
            return
        
        active_segments = self.digit_segments[char]
        dim_color = tuple(int(c * 0.15) for c in fg_color)  # 15% brightness for inactive
        
        # Draw all segments
        for seg_name, seg_coords in self.segments.items():
            color = fg_color if seg_name in active_segments else (dim_color if dim_inactive else bg_color)
            
            # Scale coordinates to digit size
            scaled_coords = [
                (x + int(px * self.digit_width), y + int(py * self.digit_height))
                for px, py in seg_coords
            ]
            draw.polygon(scaled_coords, fill=color)
        
        # Draw decimal point if needed
        if char == '.':
            point_x = x + int(self.digit_width * 0.5)
            point_y = y + int(self.digit_height * 0.90)
            point_radius = int(self.thickness * 1.2)
            draw.ellipse(
                [point_x - point_radius, point_y - point_radius,
                 point_x + point_radius, point_y + point_radius],
                fill=fg_color
            )

# ==================== EFFECTS AND AUGMENTATION ====================
def add_bezel_frame(img, text_label="kWh METER"):
    """Add realistic meter bezel/frame with surrounding text"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Draw bezel border
    border_color = random.choice([(80, 80, 80), (60, 60, 60), (100, 100, 100)])
    border_width = random.randint(5, 12)
    draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=border_width)
    
    # Add corner screws
    screw_radius = random.randint(3, 6)
    screw_color = (50, 50, 50)
    corners = [(15, 15), (width-15, 15), (15, height-15), (width-15, height-15)]
    for cx, cy in corners:
        draw.ellipse([cx-screw_radius, cy-screw_radius, cx+screw_radius, cy+screw_radius], 
                     fill=screw_color)
        draw.ellipse([cx-screw_radius//2, cy-screw_radius//2, cx+screw_radius//2, cy+screw_radius//2], 
                     fill=(30, 30, 30))
    
    # Add text label (top or bottom)
    try:
        font_size = random.randint(10, 14)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text_color = (200, 200, 200)
    position = random.choice(["top", "bottom"])
    
    if position == "top":
        text_y = 8
    else:
        text_y = height - 20
    
    # Center the text
    bbox = draw.textbbox((0, 0), text_label, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (width - text_width) // 2
    
    draw.text((text_x, text_y), text_label, fill=text_color, font=font)
    
    return img

def add_branding_text(img):
    """Add realistic meter branding (manufacturer, property labels, etc.)"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    brands = [
        "PROPERTY OF UTILITY CO.",
        "TANGEDCO METER",
        "ELECTRIC METER",
        "SMART METER",
        "kWh ENERGY",
    ]
    
    try:
        font_size = random.randint(8, 12)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text = random.choice(brands)
    text_color = (150, 150, 150)
    
    # Random position (avoid center display area)
    if random.random() > 0.5:
        # Top position
        text_y = random.randint(5, 15)
    else:
        # Bottom position
        text_y = height - random.randint(15, 25)
    
    text_x = random.randint(20, 50)
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    
    return img

def add_lighting_effects(img, style="random"):
    """Add realistic lighting: shadows, highlights, vignette"""
    width, height = img.size
    
    # Create lighting overlay
    lighting = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(lighting)
    
    if style == "random":
        style = random.choice(["gradient", "spotlight", "shadow", "none"])
    
    if style == "gradient":
        # Vertical gradient (brighter at top)
        for y in range(height):
            brightness = int(255 * (1 - y / height * 0.3))
            draw.rectangle([0, y, width, y+1], fill=(brightness, brightness, brightness))
    
    elif style == "spotlight":
        # Radial gradient from center
        center_x, center_y = width // 2, height // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                brightness = int(255 * (1 - dist / max_dist * 0.4))
                draw.point((x, y), fill=(brightness, brightness, brightness))
    
    elif style == "shadow":
        # Corner shadow
        corner = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
        for y in range(height):
            for x in range(width):
                if corner == "top-left":
                    dist = np.sqrt(x**2 + y**2)
                elif corner == "top-right":
                    dist = np.sqrt((width-x)**2 + y**2)
                elif corner == "bottom-left":
                    dist = np.sqrt(x**2 + (height-y)**2)
                else:
                    dist = np.sqrt((width-x)**2 + (height-y)**2)
                
                max_dist = np.sqrt(width**2 + height**2)
                brightness = int(255 * (0.7 + dist / max_dist * 0.3))
                draw.point((x, y), fill=(brightness, brightness, brightness))
    
    # Blend lighting with original image
    if style != "none":
        img = Image.blend(img, lighting, alpha=0.2)
    
    return img

def add_perspective_tilt(img, max_angle=15):
    """Add perspective transformation (tilt/rotation)"""
    angle = random.uniform(-max_angle, max_angle)
    img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    return img

def add_blur_and_noise(img):
    """Add realistic blur and noise effects"""
    # Random blur
    if random.random() > 0.6:
        blur_radius = random.uniform(0.5, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Add noise
    if random.random() > 0.5:
        img_array = np.array(img)
        noise = np.random.normal(0, random.randint(5, 15), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    # Adjust brightness/contrast
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    return img

# ==================== MAIN GENERATION FUNCTION ====================
def generate_realistic_meter_image(text, style_config, apply_effects=True):
    """Generate a single realistic meter image"""
    
    # Create base image with background color
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), style_config["bg_color"])
    draw = ImageDraw.Draw(img)
    
    # Calculate layout for 7-segment display
    display = SevenSegmentDisplay(
        digit_width=random.randint(35, 45),
        digit_height=random.randint(70, 90),
        segment_thickness=random.randint(6, 10),
        spacing=random.randint(5, 15)
    )
    
    # Calculate total width and center position
    num_digits = len(text)
    total_width = num_digits * (display.digit_width + display.spacing)
    start_x = (IMAGE_WIDTH - total_width) // 2
    start_y = (IMAGE_HEIGHT - display.digit_height) // 2
    
    # Draw each digit
    for i, char in enumerate(text):
        x = start_x + i * (display.digit_width + display.spacing)
        y = start_y
        
        if char == '.':
            # Decimal point is narrower
            x -= display.digit_width // 3
        
        display.draw_digit(draw, x, y, char, 
                          fg_color=style_config["fg_color"],
                          bg_color=style_config["bg_color"],
                          dim_inactive=random.random() > 0.3)  # 70% show dim segments
    
    # Apply realistic effects
    if apply_effects:
        # Add bezel/frame (60% chance)
        if random.random() > 0.4:
            labels = ["kWh METER", "ENERGY METER", "ELECTRIC", "SMART METER"]
            img = add_bezel_frame(img, text_label=random.choice(labels))
        
        # Add branding text (40% chance)
        if random.random() > 0.6:
            img = add_branding_text(img)
        
        # Add lighting effects (70% chance)
        if random.random() > 0.3:
            img = add_lighting_effects(img)
        
        # Add perspective tilt (30% chance)
        if random.random() > 0.7:
            img = add_perspective_tilt(img, max_angle=10)
        
        # Add blur and noise (50% chance)
        if random.random() > 0.5:
            img = add_blur_and_noise(img)
    
    return img

# ==================== DATASET GENERATION ====================
def generate_dataset():
    """Generate complete dataset with validation CSV"""
    
    print(f"🎨 REALISTIC METER FACE SYNTHETIC DATA GENERATOR")
    print(f"=" * 60)
    print(f"📊 Generating {NUM_SAMPLES:,} realistic meter images...")
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"🎯 Character set: {CHARS}")
    print(f"=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Prepare weighted pattern selection
    pattern_choices = []
    pattern_weights = []
    for p in METER_PATTERNS:
        pattern_choices.append(p)
        pattern_weights.append(p["weight"])
    
    # Prepare weighted style selection
    style_choices = []
    style_weights = []
    for name, config in VISUAL_STYLES.items():
        style_choices.append(config)
        style_weights.append(config["weight"])
    
    # Generate dataset
    validation_data = []
    preview_images = []
    
    for i in tqdm(range(NUM_SAMPLES), desc="Generating images"):
        # Select pattern and generate text
        pattern = random.choices(pattern_choices, weights=pattern_weights)[0]
        text = pattern["pattern"]()
        
        # Select visual style
        style = random.choices(style_choices, weights=style_weights)[0]
        
        # Generate image
        img = generate_realistic_meter_image(text, style, apply_effects=True)
        
        # Save image
        filename = f"meter_{i:06d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        img.save(filepath)
        
        # Record metadata
        validation_data.append({
            "filename": filename,
            "text": text,
            "pattern_type": pattern["type"]
        })
        
        # Collect preview samples
        if i < PREVIEW_SAMPLES:
            preview_images.append((img, text))
    
    # Save validation CSV
    csv_path = os.path.join(OUTPUT_DIR, "validation.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "pattern_type"])
        writer.writeheader()
        writer.writerows(validation_data)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Output: {OUTPUT_DIR}/")
    print(f"📄 Validation CSV: {csv_path}")
    print(f"📊 Total samples: {NUM_SAMPLES:,}")
    
    # Generate preview grid
    generate_preview_grid(preview_images)
    
    # Print statistics
    print_statistics(validation_data)

def generate_preview_grid(preview_images):
    """Create a preview grid of sample images"""
    rows, cols = 4, 5
    cell_width, cell_height = IMAGE_WIDTH, IMAGE_HEIGHT
    grid_width = cols * cell_width
    grid_height = rows * cell_height
    
    grid = Image.new('RGB', (grid_width, grid_height), (40, 40, 40))
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, text) in enumerate(preview_images[:rows*cols]):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = row * cell_height
        
        grid.paste(img, (x, y))
        
        # Add text label overlay
        draw = ImageDraw.Draw(grid)
        label_bg = [(x, y + cell_height - 25), (x + cell_width, y + cell_height)]
        draw.rectangle(label_bg, fill=(0, 0, 0, 180))
        draw.text((x + 10, y + cell_height - 20), f"GT: {text}", fill=(255, 255, 0), font=font)
    
    preview_path = os.path.join(OUTPUT_DIR, "preview_grid.png")
    grid.save(preview_path)
    print(f"🖼️  Preview grid: {preview_path}")

def print_statistics(validation_data):
    """Print dataset statistics"""
    print(f"\n📈 DATASET STATISTICS:")
    print(f"─" * 60)
    
    # Count patterns
    pattern_counts = {}
    for item in validation_data:
        pt = item["pattern_type"]
        pattern_counts[pt] = pattern_counts.get(pt, 0) + 1
    
    print(f"Pattern distribution:")
    for pattern, count in sorted(pattern_counts.items()):
        percentage = count / len(validation_data) * 100
        print(f"  {pattern:12s}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\n✨ Sample labels:")
    samples = random.sample(validation_data, min(10, len(validation_data)))
    for item in samples:
        print(f"  {item['filename']:20s} → {item['text']:12s} ({item['pattern_type']})")
    
    print(f"─" * 60)
    print(f"\n🚀 NEXT STEP: Train the model")
    print(f"   python train_on_realistic.py")

# ==================== MAIN ====================
if __name__ == "__main__":
    generate_dataset()
