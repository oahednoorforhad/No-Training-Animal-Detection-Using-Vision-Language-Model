import cv2
import torch
import time
from groundingdino.util.inference import Model
from groundingdino.util import box_ops
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")


# ---------- Load GroundingDINO ----------
# dino_model = Model(
#     model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
#     model_checkpoint_path="GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
# )

# ---------- Load LLaVA ----------
# Use a quantized model for smaller GPUs
llava_model_id = "unsloth/llava-1.5-7b-hf-bnb-4bit"   # 4-bit quantized
# Alternative smaller one: "llava-hf/llava-1.5-3b"

processor = AutoProcessor.from_pretrained(llava_model_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_model_id,
    device_map="auto",
    load_in_4bit=True  # very important for small GPUs
)

# ---------- Video Capture ----------
cap = cv2.VideoCapture(0)
last_detection_time = 0
interval = 5  # seconds
latest_boxes, latest_phrases = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    # Run detection every 5 seconds
    current_time = time.time()
    if current_time - last_detection_time >= interval:
        result = dino_model.predict_with_caption(frame, caption="animal", box_threshold=0.3, text_threshold=0.25)
        
        if len(result) == 3:
            latest_boxes, logits, latest_phrases = result
        else:
            latest_boxes, latest_phrases = result
            logits = None
        
        # Classify each detected animal with LLaVA
        if latest_boxes is not None and len(latest_boxes) > 0:
            classified_labels = []
            
            for box_tuple in latest_boxes:
                try:
                    box_coords = box_tuple[0]
                    x1, y1, x2, y2 = map(int, box_coords)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Skip if crop is too small
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        classified_labels.append("unknown")
                        continue
                    
                    # Crop detected animal
                    crop = frame[y1:y2, x1:x2]
                    pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    
                    # LLaVA conversational format
                    prompt = "USER: <image>\nWhat animal is this? Answer with one word only.\nASSISTANT:"
                    
                    inputs = processor(
                        images=pil_image,
                        text=prompt,
                        return_tensors="pt"
                    )
                    
                    # Move to same device as model
                    inputs = {k: v.to(llava_model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        output = llava_model.generate(**inputs, max_new_tokens=20)
                    
                    # Decode only the generated part (skip the prompt)
                    answer = processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    answer = answer.strip().split()[0] if answer.strip() else "unknown"
                    
                    classified_labels.append(answer)
                    
                    # Clean up GPU memory
                    del inputs, output
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error classifying box: {e}")
                    classified_labels.append("error")
                    continue
            
            latest_phrases = classified_labels
        
        last_detection_time = current_time
    
    # ---------- Draw Results ----------
    if latest_boxes is not None and len(latest_boxes) > 0:
        for i, (box_tuple, label) in enumerate(zip(latest_boxes, latest_phrases)):
            try:
                box_coords = box_tuple[0]
                x1, y1, x2, y2 = map(int, box_coords)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(label), (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error drawing box {i}: {e}")
                continue
    
    cv2.imshow("Animal Detector (GroundingDINO + LLaVA)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()