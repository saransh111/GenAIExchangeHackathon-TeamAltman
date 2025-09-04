import os
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from byaldi import RAGMultiModalModel  # ColPali wrapper
from pdf2image import convert_from_path
from PIL import Image

# ---------- Optional VLM imports (only needed if you enable extraction) ----------
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    HAS_VLM = True
except Exception:
    HAS_VLM = False


# --------------------------- Utilities ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pdf_to_images(pdf_path: str, out_dir: str) -> List[str]:
    """
    Converts a PDF to PNG images (one per page). Returns list of image paths.
    """
    ensure_dir(out_dir)
    pages = convert_from_path(pdf_path)  # requires poppler installed
    image_paths = []
    for i, img in enumerate(pages, start=1):
        p = os.path.join(out_dir, f"page_{i:04d}.png")
        img.save(p, "PNG")
        image_paths.append(p)
    return image_paths


def load_or_init_rag(model_name: str = "vidore/colpali-v1.2", device: str = "cpu") -> RAGMultiModalModel:
    return RAGMultiModalModel.from_pretrained(
        model_name,
        device=device   # only pass device
    )


def build_index(rag: RAGMultiModalModel,
                input_path: str,
                index_name: str,
                store_images_in_index: bool = True):
    """
    Index PDFs or image files in input_path.
    - If store_images_in_index=True, search results will include base64 of pages,
      which is super handy for feeding directly to a VLM for extraction.
    """
    rag.index(
        input_path=input_path,
        index_name=index_name,
        store_collection_with_index=store_images_in_index,
        overwrite=True
    )


@dataclass
class RetrievalResult:
    doc_id: int
    page_num: int
    score: float
    base64_img: Optional[str]
    metadata: Dict[str, Any]


def search(rag, query, k=5):
    results = rag.search(query, k=k)
    hits = []
    for r in results:
        hits.append({
            "doc_id": r.doc_id,
            "page_num": r.page_num,
            "score": r.score,
            "text": r.text,
        })
    return hits


# --------------------------- Optional: Structured Extraction via VLM ---------------------------

class VisionExtractor:
    """
    Wraps Qwen2-VL for extracting structured JSON from retrieved page images.
    You can swap this with any VLM you like (Gemini, Claude, etc.).
    """
    def __init__(self,
                 model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str = "cpu"):
        if not HAS_VLM:
            raise RuntimeError("VLM dependencies not installed. Re-run pip installs above.")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        if device == "cuda":
            self.model = self.model.cuda().eval()
        self.processor = Qwen2VLProcessor.from_pretrained(model_id, min_pixels=224*224, max_pixels=1024*1024)
        self.device = device

    def _pil_from_b64(self, b64png: str) -> Image.Image:
        return Image.open(base64.b64decode(b64png))

    def extract_json(self,
                     page_images: List[Image.Image],
                     instruction: str = "Extract key facts as JSON with fields: {title, key_points, metrics, chart_types}") -> str:
        """
        Sends up to 3 images + an instruction to the VLM and asks for JSON.
        Returns VLM text output (ideally JSON).
        """
        images = page_images[:3] if len(page_images) > 3 else page_images
        chat = [{
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images] +
                [{"type": "text",
                  "text": f"{instruction}\nRules: Return VALID JSON only. "
                          f"If there are charts, include their type and any visible numbers/axes. "
                          f"If a table exists, extract header and first 5 rows."}]
            )
        }]

        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(chat)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=700)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]


def results_to_pils(results: List[RetrievalResult]) -> List[Image.Image]:
    imgs = []
    for r in results:
        if r.base64_img:
            # base64 may be "data:image/png;base64,...." or just the payload
            b64 = r.base64_img
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img = Image.open(base64.b64decode(b64))
            imgs.append(img)
    return imgs


# --------------------------- Demo main ---------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ColPali (Byaldi) pipeline for slide decks")
    parser.add_argument("--input", required=True,
                        help="Path to a PDF, a single image, or a directory of PDFs/images.")
    parser.add_argument("--index-name", default="slides_index")
    parser.add_argument("--model", default="vidore/colpali-v1.2",
                        help="Byaldi model checkpoint (e.g., vidore/colpali-v1.2 or vidore/colqwen2-v1.0).")
    parser.add_argument("--query", default="Where is the revenue growth chart discussed?")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--extract", action="store_true",
                        help="Run VLM-based structured extraction on the top-k pages.")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()
    args.input = r"C:\Users\saran\OneDrive\Desktop\GENAIEXHANGEHACKATHON\file-example_PDF_500_kB.pdf"
    # 1) If you pass a PPTX, export it to PDF first (PowerPoint -> Save as PDF).
    #    If you pass a PDF, you can index it directly – no need to pre-convert to images.

    # 2) Load ColPali (Byaldi) and build index
    rag = load_or_init_rag(args.model, device=args.device)
    build_index(
        rag,
        input_path=args.input,               # file or directory
        index_name=args.index_name,
        store_images_in_index=True           # so retrieval returns base64 images (great for VLM)
    )

    # 3) Retrieve relevant pages for your question
    hits = search(rag, query=args.query, k=args.topk)
    print("\nTop results:")
    for i, h in enumerate(hits, 1):
        print(f"{i}. doc_id={h['doc_id']} page={h['page_num']} score={h['score']:.4f}")


    # 4) Optional: ask a VLM to extract structured data from those retrieved pages
    if args.extract:
        if not HAS_VLM:
            raise SystemExit("Extraction requested but VLM deps not installed. See pip installs above.")
        vlm = VisionExtractor(device=args.device)
        pil_images = results_to_pils(hits)
        json_like = vlm.extract_json(
            pil_images,
            instruction="Extract slide title, bullet points, visible metrics/values, and chart types as JSON."
        )
        print("\n--- Structured Extraction (VLM output) ---")
        print(json_like)
