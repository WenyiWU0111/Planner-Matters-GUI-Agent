import base64
import re
from io import BytesIO

import PIL.Image as Image
from playwright.sync_api import TimeoutError as PWTimeoutError

from .actions import ActionTypes
from .grounding_model import get_coords_with_2stage_grounding


def _parse_point(coords_text: str):
    """Parse '<point>x y</point>' into (x, y) integers if present."""
    if not coords_text:
        return None
    m = re.search(r'<\s*point\s*>(\d+)\s+(\d+)\s*</\s*point\s*>', str(coords_text), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None



def _click_prefer_clickable(page, x, y):
    """
    Move/click at (x, y), preferring the nearest clickable ancestor under the point.
    If a popup opens, switch context to it; otherwise, wait briefly for SPA changes.
    """
    js = """
    ([x, y]) => {
      const el = document.elementFromPoint(x, y);
      if (!el) return null;
      const target = el.closest('a,button,[role="button"],input,textarea,select') || el;
      const r = target.getBoundingClientRect();
      return { x: Math.floor(r.left + r.width/2), y: Math.floor(r.top + r.height/2),
               tag: target.tagName, href: target.href || '' };
    }
    """
    try:
        info = page.evaluate(js, [int(x), int(y)])
    except Exception:
        info = None
    if info:
        page.mouse.move(info["x"], info["y"])
        page.mouse.click(info["x"], info["y"])
        try:
            popup = page.wait_for_event("popup", timeout=1000)
            try:
                popup.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass
            return popup
        except PWTimeoutError:
            pass
        try:
            page.wait_for_timeout(800)
        except Exception:
            pass
        return page
    # Fallback to direct click
    page.mouse.move(int(x), int(y))
    page.mouse.click(int(x), int(y))
    try:
        page.wait_for_timeout(500)
    except Exception:
        pass
    return page



def execute_pixel_action(responses, page, image_processor=None, observation=None, args=None):
    """
    Execute actions using Playwright based on the model's output
    
    Args:
        page: Playwright page object
        responses: Dictionary or list of dictionaries containing action data
        image_processor: Optional ImageObservationProcessor for visualization
        observation: Optional observation dict to update with visualization
    
    Returns:
        True if actions were executed, "DONE" if finished action was encountered
    """
    #########
    MAX_RETRIES = 3
    id2center = {}
    
    for attempt in range(MAX_RETRIES):
        try:
            # Take screenshot
            image_base64 = observation.get("image", None) if observation else None
            screenshot_img = Image.open(BytesIO(base64.b64decode(image_base64)))
            som_bboxes = image_processor.get_page_bboxes(page)
            # Process the screenshot
            bbox_img, id2center, content_str = image_processor.draw_bounding_boxes(
                som_bboxes,
                screenshot_img,
                viewport_size=image_processor.viewport_size,
            )
            # If we reach here without an exception, we succeeded
            break
        except Exception as e:
            print(f"Error on attempt {attempt+1}/{MAX_RETRIES}: {e}")
    #########################################################
    if isinstance(responses, dict):
        responses = [responses]

    grounding_mode = getattr(args, 'grounding_mode', 'auto') if args else 'auto'
    for response_id, response in enumerate(responses):
        # Extract action data from new action structure
        action_type = response.get("action_type")
        
        print(f"Executing {action_type} action...")

        if action_type == ActionTypes.CLICK:
            # Handle click action with description
            description = response.get("description", "").lower()
            reasoning = response.get("reasoning", "")
            element_id = response.get("element_id", "")
            element_id = element_id[0] if isinstance(element_id, list) else element_id
            coords_str = response.get("coords", "")
            # Prefer provided pixel coords if present (unless forcing grounding)
            # pt = _parse_point(coords_str)
            pt = None
            if grounding_mode != 'force' and pt:
                page.mouse.move(pt[0], pt[1])
                page.mouse.click(pt[0], pt[1])
                print(f"Clicked at provided coordinates {pt} for element: {description}")
                continue
            # Decide ordering based on grounding_mode
            used_direct_center = False
            if grounding_mode in ('auto', 'off'):
                # Next, prefer element center if available (only if element_id is numeric and in id2center)
                if image_processor is not None and element_id and element_id in id2center:
                    try:
                        center = image_processor.get_element_center(element_id)
                        vp = page.viewport_size
                        final_coords = (center[0] * vp["width"], center[1] * vp["height"])
                        if final_coords != (0.0, 0.0):  # Validate we got a real center
                            page = _click_prefer_clickable(page, final_coords[0], final_coords[1])
                            print(f"Clicked at element center {final_coords} for element_id={element_id} ({description})")
                            used_direct_center = True
                            # Set interaction point for visualization if image_processor is available
                            if image_processor is not None:
                                image_processor.set_interaction_point(final_coords[0], final_coords[1])
                    except Exception:
                        pass
                if used_direct_center:
                    continue
                if grounding_mode == 'off':
                    # Grounding disabled entirely
                    print("Grounding mode is 'off' and no direct center/coords available; skipping grounding.")
                    continue
            # In 'prefer' mode we attempt grounding before using element center
            if grounding_mode == 'prefer':
                pass  # fall through to grounding call below
            # Must use grounding model to get coordinates with 2-stage grounding
            if not args or not hasattr(args, 'grounding_model') or not args.grounding_model:
                print(f"Error: grounding_model is required for click action but not available")
                final_coords = [0, 0]
            else:
                image_base64 = observation.get("image", None) if observation else None
                coords, element_id, success = get_coords_with_2stage_grounding(
                    coords_str, description, args.grounding_model, image_base64, id2center, 
                    max_distance=50, max_attempts=3
                )
                
                if success and element_id is not None:
                    # Use the matched element's center
                    element_center = image_processor.get_element_center(element_id) 
                    viewport_size = page.viewport_size
                    final_coords = (element_center[0] * viewport_size["width"], element_center[1] * viewport_size["height"])
                else:
                    # Use the coordinates directly (might not match an element)
                    viewport_size = page.viewport_size
                    final_coords = (coords[0], coords[1]) if coords else (0, 0)
                    print(f"Warning: Could not match element, using raw coordinates: {final_coords}")
            
            page = _click_prefer_clickable(page, final_coords[0], final_coords[1])
            print(f"Clicked at coordinates {final_coords} for element: {description}")
            # Set interaction point for visualization if image_processor is available
            if image_processor is not None:
                image_processor.set_interaction_point(final_coords[0], final_coords[1])

        elif action_type == ActionTypes.SELECT:
            # Handle select action
            element_id = response.get("element_id", "")
            element_id = element_id[0] if isinstance(element_id, list) else element_id
            description = response.get("description", "")
            reasoning = response.get("reasoning", "")
            coords_str = response.get("coords", "")
            field_description = f"the selection bar of the dropdown menu, and select {description} from it"
            # Prefer provided coords (unless forcing grounding)
            # pt = _parse_point(coords_str)
            pt = None
            if grounding_mode != 'force' and pt:
                page.mouse.move(pt[0], pt[1])
                page.mouse.click(pt[0], pt[1])
                print(f"Clicked at provided coordinates {pt} for element: {description}")
                continue
            used_direct_center = False
            if grounding_mode in ('auto', 'off'):
                # Prefer element center if available (only if element_id is numeric and in id2center)
                if image_processor is not None and element_id and element_id in id2center:
                    try:
                        center = image_processor.get_element_center(element_id)
                        vp = page.viewport_size
                        final_coords = (center[0] * vp["width"], center[1] * vp["height"])
                        if final_coords != (0.0, 0.0):  # Validate we got a real center
                            page = _click_prefer_clickable(page, final_coords[0], final_coords[1])
                            print(f"Clicked at element center {final_coords} for selection: {description}")
                            used_direct_center = True
                            # Set interaction point for visualization if image_processor is available
                            if image_processor is not None:
                                image_processor.set_interaction_point(final_coords[0], final_coords[1])
                    except Exception:
                        pass
                if used_direct_center:
                    continue
                if grounding_mode == 'off':
                    print("Grounding mode is 'off' and no direct center/coords available; skipping grounding.")
                    continue
            if not args or not hasattr(args, 'grounding_model') or not args.grounding_model:
                print(f"Error: grounding_model is required for select action but not available")
                final_coords = [0, 0]
            else:
                image_base64 = observation.get("image", None) if observation else None
                coords, element_id, success = get_coords_with_2stage_grounding(
                    coords_str, field_description, args.grounding_model, image_base64, id2center, 
                    max_distance=50, max_attempts=3
                )
                
                if success and element_id is not None:
                    # Use the matched element's center
                    element_center = image_processor.get_element_center(element_id) 
                    viewport_size = page.viewport_size
                    final_coords = (element_center[0] * viewport_size["width"], element_center[1] * viewport_size["height"])
                else:
                    # Use the coordinates directly (might not match an element)
                    viewport_size = page.viewport_size
                    final_coords = (coords[0], coords[1]) if coords else (0, 0)
                    print(f"Warning: Could not match element, using raw coordinates: {final_coords}")
            
            page = _click_prefer_clickable(page, final_coords[0], final_coords[1])
            print(f"Clicked at coordinates {final_coords} for element: {description}")
            # Set interaction point for visualization if image_processor is available
            if image_processor is not None:
                image_processor.set_interaction_point(final_coords[0], final_coords[1])
        elif action_type == ActionTypes.TYPE:
            # Handle type action with field description
            element_id = response.get("element_id", "")
            element_id = element_id[0] if isinstance(element_id, list) else element_id
            text = response.get("text", "")
            field_description = response.get("field_description", "search input field")
            reasoning = response.get("reasoning", "")
            coords_str = response.get("coords", "")
            get_coords = False
            
            search_urls = {
                'wiki': 'https://en.wikipedia.org/w/index.php?search={}&title=Special%3ASearch&ns0=1',
                'allrecipes': 'https://www.allrecipes.com/search?q={}',
                'amazon': 'https://www.amazon.com/s?k={}'
            }
            
            # Find which site we're on
            for site_key, url_template in search_urls.items():
                if site_key in page.url:
                    # Format the search URL with the query text
                    target_url = url_template.format(text.replace(' ', '+'))
                    try:
                        page.goto(target_url)
                        return page
                    except Exception:
                        pass
            else:
                # Prefer provided coords (unless forcing grounding)
                # pt = _parse_point(coords_str)
                pt = None
                if grounding_mode != 'force' and pt:
                    page.mouse.move(pt[0], pt[1])
                    page.mouse.click(pt[0], pt[1])
                    print(f"Clicked at provided coordinates {pt} for element: {field_description}")
                    # Clear and type
                    for _ in range(50):
                        page.keyboard.press("Delete")
                    page.keyboard.type(text)
                    page.wait_for_timeout(1000)
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(2000)
                    print(f"Typed '{text}' at coordinates {pt} for field: {field_description}")
                    continue
                used_direct_center = False
                if grounding_mode in ('auto', 'off'):
                    # Prefer element center if available (only if element_id is numeric and in id2center)
                    if image_processor is not None and element_id and element_id in id2center:
                        try:
                            center = image_processor.get_element_center(element_id)
                            vp = page.viewport_size
                            final_coords = (center[0] * vp["width"], center[1] * vp["height"])
                            if final_coords != (0.0, 0.0):  # Validate we got a real center
                                page = _click_prefer_clickable(page, final_coords[0], final_coords[1])
                                print(f"Clicked at element center {final_coords} for field: {field_description}")
                                for _ in range(50):
                                    page.keyboard.press("Delete")
                                page.keyboard.type(text)
                                page.wait_for_timeout(1000)
                                page.keyboard.press("Enter")
                                page.wait_for_timeout(2000)
                                print(f"Typed '{text}' at element center {final_coords} for field: {field_description}")
                                used_direct_center = True
                                # Set interaction point for visualization if image_processor is available
                                if image_processor is not None:
                                    image_processor.set_interaction_point(final_coords[0], final_coords[1])
                        except Exception:
                            pass
                    if used_direct_center:
                        continue
                    if grounding_mode == 'off':
                        print("Grounding mode is 'off' and no direct center/coords available; skipping grounding.")
                        continue
                # Fallback: use grounding model to get coordinates with 2-stage grounding
                if not args or not hasattr(args, 'grounding_model') or not args.grounding_model:
                    print(f"Error: grounding_model is required for type action but not available")
                    final_coords = [0, 0]
                else:
                    image_base64 = observation.get("image", None) if observation else None
                    coords, element_id, success = get_coords_with_2stage_grounding(
                        coords_str, field_description, args.grounding_model, image_base64, id2center, 
                        max_distance=50, max_attempts=3
                    )
                    
                    if success and element_id is not None:
                        # Use the matched element's center
                        element_center = image_processor.get_element_center(element_id) 
                        viewport_size = page.viewport_size
                        final_coords = (element_center[0] * viewport_size["width"], element_center[1] * viewport_size["height"])
                    else:
                        # Use the coordinates directly (might not match an element)
                        viewport_size = page.viewport_size
                        final_coords = (coords[0], coords[1]) if coords else (0, 0)
                        print(f"Warning: Could not match element, using raw coordinates: {final_coords}")
                
                page = _click_prefer_clickable(page, final_coords[0], final_coords[1])
                print(f"Clicked at coordinates {final_coords} for element: {field_description}")
                # Set interaction point for visualization if image_processor is available
                if image_processor is not None:
                    image_processor.set_interaction_point(final_coords[0], final_coords[1])
                # Clear the field and type the text
                # page.keyboard.press("Control+a")
                for _ in range(50):
                    page.keyboard.press("Delete")
                page.keyboard.type(text)
                page.wait_for_timeout(1000)
                page.keyboard.press("Enter")
                page.wait_for_timeout(2000)
                print(f"Typed '{text}' at coordinates {coords} for field: {field_description}")
                
        elif action_type == ActionTypes.SCROLL:
            # Handle scroll action
            direction = response.get("direction", "down")
            reasoning = response.get("reasoning", "")
            
            if direction == "up":
                page.mouse.wheel(0, -500)
            elif direction == "down":
                page.mouse.wheel(0, 500)
            elif direction == "left":
                page.mouse.wheel(-500, 0)
            elif direction == "right":
                page.mouse.wheel(500, 0)
            
            print(f"Scrolled {direction}")

        elif action_type == ActionTypes.WAIT:
            # Handle wait action
            seconds = response.get("seconds", 2.0)
            reasoning = response.get("reasoning", "")
            
            page.wait_for_timeout(int(seconds * 1000))
            print(f"Waited for {seconds} seconds")

        elif action_type == ActionTypes.KEY_PRESS:
            # Handle key press action
            key_comb = response.get("key_comb", "enter")
            reasoning = response.get("reasoning", "")
            
            page.keyboard.press(key_comb)
            print(f"Pressed key: {key_comb}")
        
        elif action_type == ActionTypes.GO_BACK:
            # Handle go back navigation
            try:
                # page.go_back(wait_until="domcontentloaded")
                page.goto(response.get("url", ""))
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
                print("Navigated back using history API")
            except Exception:
                # Fallback to keyboard shortcut
                try:
                    page.keyboard.press("Alt+ArrowLeft")
                except Exception:
                    try:
                        page.keyboard.press("Meta+ArrowLeft")
                    except Exception:
                        pass
                try:
                    page.wait_for_timeout(800)
                except Exception:
                    pass
            return page

        elif action_type == ActionTypes.GOTO_URL:
            # Handle goto URL navigation
            url = response.get("url", "")
            if url:
                try:
                    page.goto(url)
                    
                    
                    print(f"Navigated to URL: {url}")
                except Exception as e:
                    print(f"Failed to navigate to {url}: {e}")
            else:
                print("No URL provided for GOTO_URL action")

        elif action_type == ActionTypes.STOP:
            # Handle stop action - check for 'answer', 'reason', or 'reasoning' fields
            answer = response.get("answer") or response.get("reason") or response.get("reasoning") or "Task completed"
            reasoning = response.get("reasoning", "")
            
            print(f"Task finished: {answer}")
            return "DONE"

        elif action_type == ActionTypes.VERIFIER:
            # Handle verifier action
            verification_type = response.get("verification_type", "both")
            reasoning = response.get("reasoning", "")
            print(f"Verifier action: {verification_type}")
            return page
        
        else:
            print(f"Unknown action type: {action_type}")
        
        # Add a small delay between actions
        if response_id < len(responses) - 1:
            page.wait_for_timeout(1000)  # 1 second delay
            
    return page


def get_action_description(action) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    Updated to work with the new action structure and ActionTypes enum."""

    action_type = action.get("action_type", "unknown")
    
    if action_type == ActionTypes.CLICK:
        element_id = action.get("element_id", "")
        description = action.get("description", "")
        reasoning = action.get("reasoning", "")
        action_str = f"click(description='{description}', element_id='{element_id}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.TYPE:
        element_id = action.get("element_id", "")
        text = action.get("text", "")
        field_description = action.get("field_description", "")
        reasoning = action.get("reasoning", "")
        # Escape content for proper display
        text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
        action_str = f"type(text='{text}', field='{field_description}', element_id='{element_id}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.SCROLL:
        direction = action.get("direction", "down")
        reasoning = action.get("reasoning", "")
        action_str = f"scroll(direction='{direction}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.WAIT:
        seconds = action.get("seconds", 2.0)
        reasoning = action.get("reasoning", "")
        action_str = f"wait(seconds={seconds})"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.KEY_PRESS:
        key_comb = action.get("key_comb", "enter")
        reasoning = action.get("reasoning", "")
        action_str = f"press_key(key='{key_comb}')"
        if reasoning:
            action_str += f" - {reasoning}"

    elif action_type == ActionTypes.GOTO_URL:
        url = action.get("url", "")
        reasoning = action.get("reasoning", "")
        action_str = f"goto_url(url='{url}')"
        if reasoning:
            action_str += f" - {reasoning}"

    elif action_type == ActionTypes.STOP:
        answer = action.get("answer", "")
        reasoning = action.get("reasoning", "")
        # Escape content for proper display
        answer = answer.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
        action_str = f"finished(answer='{answer}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.SELECT:
        element_id = action.get("element_id", "")
        description = action.get("description", "")
        text = action.get("text", "")
        reasoning = action.get("reasoning", "")
        action_str = f"select(description='{description}', element_id='{element_id}', text='{text}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    else:
        # For any other action types, use the new structure if available
        if "reasoning" in action:
            action_str = f"{action_type}({', '.join([f'{k}={v}' for k, v in action.items() if k not in ['action_type', 'reasoning']])}) - {action.get('reasoning', '')}"
        else:
            # Fallback to old structure
            action_str = f"{action_type}({', '.join([f'{k}={v}' for k, v in action.get('action_inputs', {}).items()])})"
        
    return action_str



