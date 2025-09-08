import argparse
import os
import sys
from io import BytesIO
from datetime import datetime

from PIL import Image  # pip install pillow
from google import genai  # pip install google-genai

PROMPT_TEMPLATE = (
    "Use the provided reference image. "
    "Focus on generating a high-fidelity 3D model of the {object_name} of the image in the style of a “3D-printed architecture model.” "
    "Preserve the object's shape, proportions, and detailed surface textures (such as grain, patterns, and material imperfections), lightly stylized for a game. "
    "Ensure that the render shows clear and realistic textures, not smooth placeholders. "
    "Use physically-based lighting and shadows to highlight material depth. "
    "Show a 45° top-down (isometric) view to emphasize dimension. "
    "Define materials precisely—reflective glass, metallic surfaces, concrete, wood—so the model reads as textured, high-quality, game-engine-ready render. "
    "Pure white background."
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Gemini (Nano Banana) image-to-image generation via Gemini API."
    )
    p.add_argument("--image", required=True, help="Path to the reference image (jpg/png/webp...).")
    p.add_argument("--object_name", required=True, help="Object name to fill into the prompt.")
    p.add_argument("--out", default=None, help="Output image filename (PNG). Default: auto timestamp.")
    return p.parse_args()

def main():
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Please set GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    # Load reference image (Pillow object is accepted by the SDK)
    try:
        ref_img = Image.open(args.image)
    except Exception as e:
        print(f"ERROR: Failed to open image: {e}", file=sys.stderr)
        sys.exit(1)

    prompt = PROMPT_TEMPLATE.format(object_name=args.object_name)

    # Create client (key can be in env; no args uses default env config)
    client = genai.Client(api_key=api_key)

    # Call the Nano Banana (Gemini 2.5 Flash Image) model for image editing
    # Official docs example uses: model="gemini-2.5-flash-image-preview" and contents=[prompt, image]
    # https references: Image editing (text+image-to-image). 
    model_id = "gemini-2.5-flash-image-preview"

    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=[prompt, ref_img],
        )
    except Exception as e:
        print(f"ERROR: Gemini API call failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract first returned image bytes and save as PNG
    image_bytes = None
    try:
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                image_bytes = part.inline_data.data
                break
    except Exception as e:
        print(f"ERROR: Unexpected response format: {e}", file=sys.stderr)
        sys.exit(1)

    if not image_bytes:
        print("ERROR: No image returned from the model.", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or f"nano_banana_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    try:
        img = Image.open(BytesIO(image_bytes))
        img.save(out_path, format="PNG")
    except Exception as e:
        print(f"ERROR: Failed to save output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Done. Saved: {out_path}")

if __name__ == "__main__":
    main()
