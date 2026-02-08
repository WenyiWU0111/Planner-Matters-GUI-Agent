import base64
import io

def clean_url(url: str) -> str:
    """Normalize URL for comparison (strip trailing slash)."""
    if url is None:
        return ""
    url = str(url).strip()
    return url[:-1] if url.endswith("/") else url

def clean_answer(answer: str) -> str:
    """Normalize answer for comparison: strip quotes and lower-case."""
    if answer is None:
        return ""
    answer = str(answer).strip("'").strip('"').lower()
    return answer


def encode_image(image) -> str:
    """Convert a PIL Image to base64 JPEG string. Returns empty string if invalid."""
    if image is None:
        return ""
    try:
        if getattr(image, "mode", "") == "RGBA":
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return ""