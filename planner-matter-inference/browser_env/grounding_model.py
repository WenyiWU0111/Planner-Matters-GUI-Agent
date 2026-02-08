import re
import json
import base64
from io import BytesIO
from PIL import Image

def get_coords_from_grounding_model(element, grounding_model, image):
    print(f"Grounding model name: {grounding_model.model_name}")
    if 'ui-ins' in grounding_model.model_name.lower():
        # Ask UI-Ins to return ONLY pixel coordinates in [x, y] form
        instruct = (
            "You are a GUI grounding model. Given a screenshot and a target element description, "
            "return ONLY the pixel coordinates of the point to click as a JSON array [x, y]. "
            "Do not return any other text, labels, or fields."
        )
        query = f"Target element description: {element}\nReturn only: [x, y]"
    elif 'ui-tars' in grounding_model.model_name.lower():
        instruct = "You are a grounding model, given the screenshot and the target element description, you need to identify the coordinates of the given element and return them in the format of click(point='<point>x1 y1</point>')."
        query = "Target element description: " + element + "\nWhat's the coordinates of the target element in the screenshot? You should return as click(point='<point>x1 y1</point>')"
    
    try:
        messages=[
            {"role": "system", "content": instruct},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
            ]}
        ]
        response, _, _ = grounding_model.chat(messages=messages, stream=False, max_tokens=32)
        response_text = response.content
        print(f"Grounding model response: {response_text}")
        
    except Exception as e:
        print(f"Error calling grounding model: {e}")
        return None
    
    if 'ui-tars' in grounding_model.model_name.lower():
        coordinates = extract_coordinates_uitars(response_text)
    elif 'ui-ins' in grounding_model.model_name.lower():
        # Support both UI-Ins-7B and UI-Ins-32B with tolerant extraction
        coordinates = extract_coordinates_uiins32b(response_text)
    else:
        coordinates = None
    return coordinates

def extract_coordinates_uitars(response_text):
    coords = re.sub(r'[^\d\s,.-]', '', response_text).strip()
    coords = re.split(r'[,\s]+', coords)
    coordinates = []
    if 'x1' in response_text:
        coords[0] = coords[0][1:]
    if 'y1' in response_text:
        coords[1] = coords[1][1:]
        print(f"Extracted coordinates with x1: {coords}")
    for coord in coords:
        if coord:
            try:
                coordinates.append(float(coord))
            except:
                print(f"Invalid coordinate value: {coord}")
                coordinates.append(0)
    coordinates = coordinates[:2]
    return coordinates

def extract_coordinates_uiins32b(response_text):
    """
    Try multiple patterns to extract pixel coordinates:
      1) [x, y] or [x1, y1, x2, y2] (bbox → center)
      2) JSON object containing "coords"/"coordinate": [x, y] or "bbox": [x1,y1,x2,y2] or keys x1,y1,x2,y2
      3) <point>x y</point> 
    Returns [x, y] on success, otherwise None.
    """
    if not response_text:
        return None
    # 1) Try [x1, y1, x2, y2] (bbox) first, then [x, y]
    m4 = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', response_text)
    if m4:
        try:
            x1 = int(m4.group(1)); y1 = int(m4.group(2)); x2 = int(m4.group(3)); y2 = int(m4.group(4))
            cx = int(round((x1 + x2) / 2))
            cy = int(round((y1 + y2) / 2))
            return [cx, cy]
        except Exception:
            pass
    m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', response_text)
    if m:
        try:
            return [int(m.group(1)), int(m.group(2))]
        except Exception:
            pass
    # 2) JSON object with coords/coordinate/bbox or x1,y1,x2,y2
    try:
        # Find first JSON-like object and parse
        start = response_text.find('{')
        if start != -1:
            depth = 0
            end = -1
            for idx in range(start, len(response_text)):
                ch = response_text[idx]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if (depth == 0):
                        end = idx + 1
                        break
            if end != -1:
                obj = json.loads(response_text[start:end])
                if isinstance(obj, dict):
                    # coords or coordinate → [x, y]
                    for key in ("coords", "coordinate"):
                        val = obj.get(key)
                        if isinstance(val, list) and len(val) >= 2:
                            try:
                                return [int(val[0]), int(val[1])]
                            except Exception:
                                pass
                    # bbox → [x1,y1,x2,y2]
                    bbox = obj.get("bbox")
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        try:
                            x1, y1, x2, y2 = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                            cx = int(round((x1 + x2) / 2))
                            cy = int(round((y1 + y2) / 2))
                            return [cx, cy]
                        except Exception:
                            pass
                    # x1,y1,x2,y2 keys
                    if all(k in obj for k in ("x1", "y1", "x2", "y2")):
                        try:
                            x1 = int(obj["x1"]); y1 = int(obj["y1"]); x2 = int(obj["x2"]); y2 = int(obj["y2"])
                            cx = int(round((x1 + x2) / 2))
                            cy = int(round((y1 + y2) / 2))
                            return [cx, cy]
                        except Exception:
                            pass
    except Exception:
        pass
    # 3) <point>x y</point>
    m2 = re.search(r'<\s*point\s*>(\d+)\s+(\d+)\s*</\s*point\s*>', response_text, flags=re.IGNORECASE)
    if m2:
        try:
            return [int(m2.group(1)), int(m2.group(2))]
        except Exception:
            pass
    return None


def crop_and_zoom_image(image_base64: str, center_x: float, center_y: float, crop_size: int = 400, zoom_scale: float = 2.0):
    """
    Crop a region around the given coordinates and zoom it to the original size.
    
    Args:
        image_base64: Base64 encoded image string
        center_x: X coordinate of the center point
        center_y: Y coordinate of the center point
        crop_size: Size of the crop region (width and height in pixels)
        zoom_scale: How much to zoom (crop will be 1/zoom_scale of original, then resized back)
    
    Returns:
        Base64 encoded cropped and zoomed image string
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_data))
        img_width, img_height = img.size
        
        # Calculate crop region (centered around the point)
        crop_half = int(crop_size / (2 * zoom_scale))
        left = max(0, int(center_x - crop_half))
        top = max(0, int(center_y - crop_half))
        right = min(img_width, int(center_x + crop_half))
        bottom = min(img_height, int(center_y + crop_half))
        
        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))
        
        # Resize back to crop_size (zoom effect)
        zoomed_img = cropped_img.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
        
        # Convert back to base64
        buffered = BytesIO()
        zoomed_img.save(buffered, format="PNG")
        zoomed_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return zoomed_base64, (left, top, right, bottom)
    except Exception as e:
        print(f"Error in crop_and_zoom_image: {e}")
        return image_base64, (0, 0, img_width, img_height)


def convert_zoomed_coords_to_original(zoomed_coords, crop_region, original_size, zoom_scale: float = 2.0):
    """
    Convert coordinates from zoomed image back to original image coordinates.
    
    Args:
        zoomed_coords: (x, y) coordinates in the zoomed image (0 to crop_size)
        crop_region: (left, top, right, bottom) of the cropped region in original image
        original_size: (width, height) of the original image
        zoom_scale: Zoom scale used
    
    Returns:
        (x, y) coordinates in the original image
    """
    zoomed_x, zoomed_y = zoomed_coords[0], zoomed_coords[1]
    left, top, right, bottom = crop_region
    orig_width, orig_height = original_size
    
    # Crop region size in original image
    crop_width = right - left
    crop_height = bottom - top
    
    # The zoomed image is crop_size x crop_size (400x400)
    # The actual crop region in original image is crop_width x crop_height
    # The zoomed image was resized from the cropped region, so the scale factor is:
    crop_size = 400  # Same as in crop_and_zoom_image
    
    # Scale factor from zoomed image to original crop region
    # Since we resized crop_width x crop_height to crop_size x crop_size:
    scale_x = crop_width / crop_size
    scale_y = crop_height / crop_size
    
    # Convert zoomed coordinates to crop region coordinates
    crop_x = zoomed_x * scale_x
    crop_y = zoomed_y * scale_y
    
    # Convert to original image coordinates
    orig_x = left + crop_x
    orig_y = top + crop_y
    
    # Clamp to original image bounds
    orig_x = max(0, min(orig_width, orig_x))
    orig_y = max(0, min(orig_height, orig_y))
    
    return [orig_x, orig_y]


def get_coords_with_2stage_grounding(
    coords_str: str,
    element_description: str,
    grounding_model,
    image_base64: str,
    id2center: dict,
    max_distance: int = 50,
    max_attempts: int = 3
):
    """
    Perform 2-stage grounding: 
    1. Get coordinates on full screen
    2. Zoom into that region and get refined coordinates
    3. Match to element, retry if not found
    
    Args:
        coords_str: Original coordinates string from action
        element_description: Description of the element to find
        grounding_model: The grounding model to use
        image_base64: Base64 encoded full screenshot
        id2center: Dictionary mapping element IDs to their centers
        max_distance: Maximum distance for matching elements
        max_attempts: Maximum number of attempts
    
    Returns:
        tuple: (coords, element_id, success) where success is True if element was matched
    """
    if not image_base64:
        return [0, 0], None, False
    
    # Strip data URL prefix if present
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]
    
    # Get original image size for coordinate conversion
    try:
        image_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_data))
        original_size = img.size
    except Exception as e:
        print(f"Error getting image size: {e}")
        return [0, 0], None, False
    
    for attempt in range(max_attempts):
        try:
            # Stage 1: Get coordinates on full screen
            stage1_coords = get_coords_from_grounding_model(
                element_description, grounding_model, image_base64
            )
            
            if not stage1_coords or len(stage1_coords) < 2:
                print(f"Stage 1 grounding failed on attempt {attempt + 1}")
                continue
            
            stage1_coords = stage1_coords[:2]
            print(f"Stage 1 coords: {stage1_coords}")
            
            # Check if we can match with stage 1 coordinates directly
            closest_item_id, distance = match_coordinates_to_item(
                stage1_coords, id2center, max_distance=0
            )
            if closest_item_id is not None:
                print(f"Matched with stage 1 coords: {closest_item_id}, distance: {distance}")
                return stage1_coords, closest_item_id, True
            
            # Stage 2: Crop and zoom around stage 1 coordinates
            zoomed_image_base64, crop_region = crop_and_zoom_image(
                image_base64, stage1_coords[0], stage1_coords[1], 
                crop_size=400, zoom_scale=2.0
            )
            
            # Get coordinates on zoomed image
            stage2_coords = get_coords_from_grounding_model(
                element_description, grounding_model, zoomed_image_base64
            )
            
            if not stage2_coords or len(stage2_coords) < 2:
                print(f"Stage 2 grounding failed on attempt {attempt + 1}")
                continue
            
            stage2_coords = stage2_coords[:2]
            print(f"Stage 2 coords (zoomed): {stage2_coords}")
            
            # Convert zoomed coordinates back to original image coordinates
            final_coords = convert_zoomed_coords_to_original(
                stage2_coords, crop_region, original_size, zoom_scale=2.0
            )
            print(f"Stage 2 coords (converted to original): {final_coords}")
            
            # Try to match with refined coordinates
            closest_item_id, distance = match_coordinates_to_item(
                final_coords, id2center, max_distance=max_distance
            )
            if closest_item_id is not None:
                print(f"Matched with stage 2 coords: {closest_item_id}, distance: {distance}")
                return final_coords, closest_item_id, True
            else:
                print(f"No match found on attempt {attempt + 1}, distance: {distance}")
        
        except Exception as e:
            print(f"Error in 2-stage grounding attempt {attempt + 1}: {e}")
            continue
    
    # If all attempts failed, return the last stage 1 coordinates (or [0, 0] if none)
    print(f"All {max_attempts} attempts failed, returning best guess coordinates")
    try:
        # Try one more time with stage 1 only as fallback
        stage1_coords = get_coords_from_grounding_model(
            element_description, grounding_model, image_base64
        )
        if stage1_coords and len(stage1_coords) >= 2:
            return stage1_coords[:2], None, False
    except:
        pass
    
    return [0, 0], None, False

def match_coordinates_to_item(coords, id2center, max_distance=None, consider_inside_first=True):
    """
    Match coordinates (x, y) to the nearest item in id2center.
    
    Args:
        coords (tuple): The (x, y) coordinates to match.
        id2center (dict): Dictionary mapping IDs to (center_x, center_y, width, height).
        max_distance (float, optional): Maximum distance for a match. If None, no limit.
        consider_inside_first (bool): If True, prioritize items that contain the point.
        
    Returns:
        tuple: (item_id, distance) of the closest item, or (None, None) if no match within max_distance.
    """
    x, y = coords[0], coords[1]
    closest_id = None
    min_distance = float('inf')
    
    # First check if the point is inside any bounding box
    inside_items = []
    if consider_inside_first:
        for item_id, (center_x, center_y, width, height) in id2center.items():
            half_width = width / 2
            half_height = height / 2
            
            # Check if point is inside this bounding box
            if (center_x - half_width <= x <= center_x + half_width and 
                center_y - half_height <= y <= center_y + half_height):
                inside_items.append((item_id, 0))  # Distance is 0 when inside
    
    # If point is inside one or more boxes, return the one with smallest area
    if inside_items:
        if len(inside_items) == 1:
            return inside_items[0]  # Return the only item containing the point
        else:
            # If multiple boxes contain the point, return the smallest one
            smallest_area = float('inf')
            smallest_id = None
            for item_id, _ in inside_items:
                _, _, width, height = id2center[item_id]
                area = width * height
                if area < smallest_area:
                    smallest_area = area
                    smallest_id = item_id
            return smallest_id, 0
    
    # If point isn't inside any box, find the closest box by Euclidean distance
    for item_id, (center_x, center_y, width, height) in id2center.items():
        # Calculate the closest point on the box to the given coordinates
        closest_x = max(center_x - width/2, min(x, center_x + width/2))
        closest_y = max(center_y - height/2, min(y, center_y + height/2))
        
        # Calculate distance to the closest point
        distance = ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_id = item_id
    
    if max_distance is not None and min_distance > max_distance:
        print(f"No item found within max distance: {max_distance}")
        return None, None
    
    return closest_id, min_distance