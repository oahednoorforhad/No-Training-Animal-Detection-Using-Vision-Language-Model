import gradio as gr
import torch
import cv2
from PIL import Image
from groundingdino.util.inference import Model
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import numpy as np
import re

# ---------- Load GroundingDINO ----------
dino_model = Model(
    model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    model_checkpoint_path="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
)

# ---------- Load LLaVA ----------
llava_model_id = "unsloth/llava-1.5-7b-hf-bnb-4bit"
llava_processor = AutoProcessor.from_pretrained(llava_model_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# ---------- Predefined gallery images ----------
gallery_folder = "images"
gallery_images = []
if os.path.exists(gallery_folder):
    gallery_images = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) 
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# ---------- Comprehensive animal validation ----------
ANIMAL_KEYWORDS = {
    # Mammals
    'dog', 'cat', 'horse', 'elephant', 'lion', 'tiger', 'bear', 'zebra', 'giraffe',
    'monkey', 'gorilla', 'chimpanzee', 'orangutan', 'wolf', 'fox', 'deer', 'moose',
    'buffalo', 'bison', 'cow', 'bull', 'cattle', 'pig', 'hog', 'sheep', 'goat', 'lamb',
    'rabbit', 'hare', 'bunny', 'squirrel', 'rat', 'mouse', 'mice', 'hamster', 'gerbil',
    'guinea', 'hedgehog', 'raccoon', 'skunk', 'otter', 'beaver', 'seal', 'walrus',
    'whale', 'dolphin', 'porpoise', 'leopard', 'cheetah', 'jaguar', 'panther', 'puma',
    'cougar', 'lynx', 'bobcat', 'hyena', 'rhino', 'rhinoceros', 'hippo', 'hippopotamus',
    'camel', 'llama', 'alpaca', 'kangaroo', 'koala', 'wombat', 'platypus', 'opossum',
    'armadillo', 'sloth', 'anteater', 'panda', 'meerkat', 'mongoose', 'weasel', 'ferret',
    'mink', 'badger', 'porcupine', 'chipmunk', 'gopher', 'mole', 'bat', 'lemur', 'baboon',
    'antelope', 'gazelle', 'impala', 'wildebeest', 'gnu', 'yak', 'ox', 'oxen', 'donkey',
    'mule', 'ass', 'stallion', 'mare', 'colt', 'foal', 'kitten', 'puppy', 'calf', 'piglet',
    'joey', 'kit', 'pup', 'cub', 'fawn', 'leveret', 'joey',
    
    # Birds
    'bird', 'eagle', 'hawk', 'falcon', 'owl', 'parrot', 'peacock', 'swan', 'duck',
    'goose', 'chicken', 'rooster', 'hen', 'chick', 'turkey', 'pigeon', 'dove', 'crow',
    'raven', 'sparrow', 'robin', 'cardinal', 'bluejay', 'jay', 'hummingbird', 'woodpecker',
    'pelican', 'flamingo', 'stork', 'crane', 'heron', 'egret', 'seagull', 'gull', 'albatross',
    'penguin', 'ostrich', 'emu', 'cassowary', 'kiwi', 'vulture', 'condor', 'kite', 'buzzard',
    'magpie', 'kingfisher', 'toucan', 'cockatoo', 'macaw', 'parakeet', 'budgie', 'canary',
    'finch', 'swallow', 'swift', 'nightingale', 'thrush', 'warbler', 'oriole', 'starling',
    'mockingbird', 'wren', 'nuthatch', 'chickadee', 'titmouse', 'pheasant', 'quail',
    'partridge', 'grouse', 'ptarmigan', 'cormorant', 'gannet', 'booby', 'frigatebird',
    'ibis', 'spoonbill', 'sandpiper', 'plover', 'curlew', 'snipe', 'avocet', 'oystercatcher',
    
    # Reptiles & Amphibians
    'snake', 'serpent', 'lizard', 'gecko', 'iguana', 'chameleon', 'turtle', 'tortoise',
    'terrapin', 'crocodile', 'alligator', 'caiman', 'komodo', 'dragon', 'cobra', 'python',
    'viper', 'rattlesnake', 'rattler', 'boa', 'anaconda', 'mamba', 'adder', 'asp',
    'frog', 'toad', 'tadpole', 'bullfrog', 'treefrog', 'salamander', 'newt', 'axolotl',
    
    # Fish & Aquatic
    'fish', 'goldfish', 'koi', 'carp', 'salmon', 'trout', 'bass', 'tuna', 'cod', 'catfish',
    'pike', 'perch', 'marlin', 'sailfish', 'swordfish', 'barracuda', 'grouper', 'snapper',
    'flounder', 'halibut', 'sole', 'turbot', 'mahi', 'mackerel', 'sardine', 'anchovy',
    'herring', 'eel', 'moray', 'ray', 'stingray', 'manta', 'skate', 'shark', 'seahorse',
    'pufferfish', 'puffer', 'clownfish', 'angelfish', 'guppy', 'betta', 'tetra', 'cichlid',
    
    # Invertebrates
    'insect', 'bug', 'butterfly', 'moth', 'bee', 'bumblebee', 'honeybee', 'wasp', 'hornet',
    'ant', 'beetle', 'ladybug', 'ladybird', 'dragonfly', 'damselfly', 'grasshopper', 'cricket',
    'mantis', 'cicada', 'fly', 'housefly', 'mosquito', 'gnat', 'midge', 'spider', 'tarantula',
    'scorpion', 'tick', 'mite', 'centipede', 'millipede', 'worm', 'earthworm', 'leech',
    'slug', 'snail', 'octopus', 'squid', 'cuttlefish', 'nautilus', 'jellyfish', 'starfish',
    'seastar', 'urchin', 'anemone', 'coral', 'crab', 'lobster', 'crayfish', 'crawfish',
    'shrimp', 'prawn', 'barnacle', 'clam', 'oyster', 'mussel', 'scallop', 'cockroach',
    'termite', 'locust', 'aphid', 'weevil', 'caterpillar', 'larvae', 'larva', 'maggot',
    'grub', 'flea', 'louse', 'bedbug', 'silverfish',
    
    # Generic terms
    'animal', 'creature', 'wildlife', 'beast', 'fauna', 'critter', 'mammal', 'reptile',
    'amphibian', 'avian', 'aquatic', 'marine', 'predator', 'prey', 'carnivore', 'herbivore'
}

NON_ANIMAL_KEYWORDS = {
    # Humans
    'human', 'person', 'man', 'woman', 'child', 'boy', 'girl', 'people', 'kid', 'baby',
    'adult', 'male', 'female', 'someone', 'individual', 'face', 'selfie', 'guy', 'lady',
    
    # Food
    'food', 'meal', 'dish', 'cuisine', 'cake', 'bread', 'pizza', 'sandwich', 'burger',
    'apple', 'banana', 'orange', 'grape', 'strawberry', 'watermelon', 'pear', 'peach',
    'plum', 'cherry', 'mango', 'pineapple', 'kiwi', 'lemon', 'lime', 'fruit', 'berry',
    'vegetable', 'carrot', 'tomato', 'potato', 'onion', 'lettuce', 'cabbage', 'broccoli',
    'meat', 'beef', 'pork', 'steak', 'cheese', 'milk', 'egg', 'butter', 'cream',
    'drink', 'beverage', 'water', 'juice', 'soda', 'coffee', 'tea', 'wine', 'beer',
    'snack', 'candy', 'chocolate', 'cookie', 'biscuit', 'dessert',
    
    # Objects
    'object', 'thing', 'item', 'stuff', 'building', 'house', 'structure', 'vehicle',
    'car', 'truck', 'bike', 'bicycle', 'motorcycle', 'furniture', 'chair', 'table',
    'desk', 'couch', 'sofa', 'bed', 'clothing', 'shirt', 'pants', 'dress', 'clothes',
    'shoe', 'toy', 'doll', 'stuffed', 'plush', 'statue', 'sculpture', 'painting',
    'picture', 'photo', 'image', 'phone', 'computer', 'screen', 'device', 'book',
    'bag', 'bottle', 'cup', 'glass', 'plate', 'bowl', 'container', 'box',
    
    # Plants
    'plant', 'tree', 'flower', 'bush', 'grass', 'vegetation', 'leaf', 'branch', 'rose',
    
    # Negative
    'nothing', 'none', 'unclear', 'unknown', 'unsure'
}

def is_animal(response_text):
    """
    Robust animal detection using multiple strategies
    Returns (is_animal: bool, label: str)
    """
    if not response_text or len(response_text.strip()) == 0:
        return False, "unknown"
    
    # Clean and normalize the response
    text_lower = response_text.lower().strip()
    text_lower = re.sub(r'[^\w\s]', ' ', text_lower)  # Remove punctuation
    text_joined = ' '.join(text_lower.split())  # Normalize spaces
    words = text_lower.split()
    
    print(f"    Analyzing response: '{response_text}'")
    print(f"    Normalized text: '{text_joined}'")
    
    # CRITICAL: Check for negative phrases FIRST (before individual words)
    negative_phrases = [
        'not an animal', 'not animal', 'no animal', 'isnt an animal', 'isnt animal',
        'is not an animal', 'is not animal', 'not a animal', 'this is not',
        'it is not', 'thats not', 'that is not'
    ]
    
    for phrase in negative_phrases:
        if phrase in text_joined:
            print(f"    ✗ REJECTED: Found negative phrase '{phrase}'")
            return False, "not_animal"
    
    # Strategy 1: Check for explicit non-animal keywords in individual words
    for word in words:
        if word in NON_ANIMAL_KEYWORDS:
            print(f"    ✗ REJECTED: Found non-animal keyword '{word}'")
            return False, word
    
    # Check for multi-word non-animal phrases (but not negative phrases already checked)
    for keyword in NON_ANIMAL_KEYWORDS:
        if ' ' in keyword and keyword in text_joined:
            print(f"    ✗ REJECTED: Found non-animal phrase '{keyword}'")
            return False, keyword
    
    # Strategy 2: Check for explicit animal keywords (accept these)
    # But ONLY if we didn't already reject with negative phrases above
    for word in words:
        if word in ANIMAL_KEYWORDS:
            print(f"    ✓ ACCEPTED: Found animal keyword '{word}'")
            return True, word
    
    # Check for multi-word animal names
    for keyword in ANIMAL_KEYWORDS:
        if ' ' in keyword and keyword in text_joined:
            print(f"    ✓ ACCEPTED: Found animal phrase '{keyword}'")
            return True, keyword
    
    # Strategy 3: If first word sounds like it could be an animal and no non-animal keywords found
    # Accept unknown animals that aren't explicitly in non-animal list
    first_word = words[0] if words else ""
    
    # Check if response is too generic or vague
    generic_responses = {'this', 'that', 'it', 'the', 'a', 'an', 'image', 'photo', 'picture'}
    if first_word in generic_responses:
        print(f"    ✗ REJECTED: Too generic response '{first_word}'")
        return False, first_word
    
    # If we got here, it's ambiguous but not explicitly non-animal
    # Accept it cautiously
    print(f"    ⚠ CAUTIOUSLY ACCEPTED: '{first_word}' (not in non-animal list)")
    return True, first_word

# ---------- Detection + Classification ----------
def detect_animals(frame):
    """Process frame and return annotated image with only animals"""
    h, w = frame.shape[:2]
    
    # Calculate text scale based on image size
    scale_factor = min(h, w) / 1080.0
    font_scale = max(0.5, 0.9 * scale_factor)
    thickness = max(1, int(2 * scale_factor))
    
    print("\n" + "="*60)
    print("Starting detection...")
    
    result = dino_model.predict_with_caption(
        frame, 
        caption="animal", 
        box_threshold=0.3, 
        text_threshold=0.25
    )

    if len(result) == 3:
        latest_boxes, _, _ = result
    else:
        latest_boxes, _ = result

    print(f"GroundingDINO found {len(latest_boxes) if latest_boxes else 0} potential detections")
    
    valid_boxes = []
    classified_labels = []
    
    if latest_boxes is not None and len(latest_boxes) > 0:
        for idx, box_tuple in enumerate(latest_boxes):
            print(f"\n--- Processing detection {idx + 1}/{len(latest_boxes)} ---")
            try:
                box_coords = box_tuple[0]
                x1, y1, x2, y2 = map(int, box_coords)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                print(f"  Box coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Skip if crop is too small
                if x2 - x1 < 20 or y2 - y1 < 20:
                    print(f"  ⊘ Skipped: Box too small")
                    continue
                
                crop = frame[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                
                # Ask LLaVA what it is
                prompt = "USER: <image>\nWhat animal is this? If it's not an animal, say 'not an animal'. Answer briefly.\nASSISTANT:"

                inputs = llava_processor(images=pil_crop, text=prompt, return_tensors="pt")
                inputs = {k: v.to(llava_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output = llava_model.generate(**inputs, max_new_tokens=25)
                
                answer = llava_processor.decode(
                    output[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                answer = answer.strip()
                
                print(f"  LLaVA response: '{answer}'")
                
                # Clean up GPU memory
                del inputs, output
                torch.cuda.empty_cache()
                
                # Validate if it's an animal
                is_valid_animal, label = is_animal(answer)
                
                if is_valid_animal:
                    valid_boxes.append(box_tuple)
                    classified_labels.append(label)
                    print(f"  ✓✓✓ FINAL: ACCEPTED as '{label}'")
                else:
                    print(f"  ✗✗✗ FINAL: REJECTED")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(valid_boxes)} animals detected out of {len(latest_boxes) if latest_boxes else 0} total detections")
    print(f"{'='*60}\n")

    # Draw boxes and labels only for valid animals
    if len(valid_boxes) > 0:
        for box_tuple, label in zip(valid_boxes, classified_labels):
            try:
                box_coords = box_tuple[0]
                x1, y1, x2, y2 = map(int, box_coords)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Draw green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                
                # Get text size for background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Draw background for text
                cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), 
                            (x1 + text_w, y1), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(frame, label, (x1, y1 - baseline - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            except Exception as e:
                print(f"Error drawing box: {e}")
                continue

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---------- Load gallery image ----------
def load_gallery_image(evt: gr.SelectData):
    """Load image from gallery folder based on selection"""
    if evt.index is not None:
        selected_path = gallery_images[evt.index]
        return Image.open(selected_path)
    return None

# ---------- Handle all three input types ----------
def process_input(input_type, uploaded_image, gallery_image, camera_image):
    """Process input based on selected type"""
    img = None
    
    if input_type == "Upload Image" and uploaded_image is not None:
        img = uploaded_image
    elif input_type == "Select from Gallery" and gallery_image is not None:
        img = gallery_image
    elif input_type == "Use Live Camera" and camera_image is not None:
        img = camera_image
    
    if img is None:
        print("ERROR: No image provided")
        return None

    # Convert to numpy array for OpenCV processing
    if isinstance(img, Image.Image):
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] == 3:
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            frame = img
    else:
        print(f"ERROR: Unexpected image type: {type(img)}")
        return None
    
    output_frame = detect_animals(frame)
    return output_frame

# ---------- Show/Hide components based on selection ----------
def update_visibility(input_type):
    """Update component visibility based on selected input type"""
    return (
        gr.update(visible=(input_type == "Upload Image")),
        gr.update(visible=(input_type == "Select from Gallery")),
        gr.update(visible=(input_type == "Use Live Camera")),
        gr.update(visible=(input_type == "Select from Gallery"))
    )

# ---------- Gradio Interface ----------
with gr.Blocks(title="Animal Detector") as demo:
    gr.Markdown("# 🐾 No Training Animal Detector")
    gr.Markdown("**GroundingDINO + LLaVa**")
    
    with gr.Row():
        input_type_selector = gr.Radio(
            ["Upload Image", "Select from Gallery", "Use Live Camera"],
            label="Select Input Type",
            value="Upload Image"
        )
    
    # Gallery selector dropdown
    gallery_preview = gr.Gallery(
        value=gallery_images if gallery_images else [],
        label="Select from Gallery (click to choose)",
        show_label=True,
        columns=4,
        rows=2,
        object_fit="contain",
        height="auto",
        visible=False,
        allow_preview=True,
        selected_index=None
    )
    
    with gr.Row():
        with gr.Column():
            upload_image_input = gr.Image(
                type="pil", 
                label="Upload Image",
                visible=True
            )
            
            gallery_input = gr.Image(
                type="pil",
                label="Selected Gallery Image",
                visible=False,
                interactive=False
            )
            
            camera_input = gr.Image(
                source="webcam",
                type="pil",
                label="Use Live Camera",
                visible=False
            )
            
            process_btn = gr.Button("🔍 Detect Animals", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Result")
    
    # Update visibility based on input type selection
    input_type_selector.change(
        fn=update_visibility,
        inputs=[input_type_selector],
        outputs=[upload_image_input, gallery_input, camera_input, gallery_preview]
    )

    # Load image when gallery image is clicked
    gallery_preview.select(
        fn=load_gallery_image,
        outputs=[gallery_input]
    )
    
    # Process button click
    process_btn.click(
        fn=process_input,
        inputs=[input_type_selector, upload_image_input, gallery_input, camera_input],
        outputs=output_image
    )
    
    gr.Markdown("""
    ### 📋 Instructions:
    1. Choose an input method (Upload, Gallery, or Camera)
    2. **For Gallery**: Select an image from the dropdown menu
    3. Select or capture your image
    4. Click "🔍 Detect Animals"
    5. **Check your terminal/console for detailed detection logs**
    """)

if __name__ == "__main__":
    demo.launch(share=True)