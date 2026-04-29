import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from io import BytesIO
import json
import time
import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from google import genai
import fitz  # PyMuPDF
from zai import ZaiClient # Zhipu AI / GLM

# Load environment variables (API Keys)
load_dotenv()

# ======================
# Base Providers (Modular approach)
# ======================

class OCRProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def run_ocr(self, image: Image.Image) -> str:
        pass

    @abstractmethod
    def run_ocr_batch(self, images: list[Image.Image]) -> list[str]:
        pass

class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def structure_data(self, ocr_text: str, pdl_context: str, callback=None) -> dict:
        pass

# ======================
# OCR Implementations
# ======================

class LightOnOCR(OCRProvider):
    @property
    def name(self):
        return "LIGHTON"

    def __init__(self, model_path="./LightOnOCR-2-1B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def load(self):
        if self.model is not None:
            return
        
        print(f"Loading LightOnOCR from {self.model_path}...")
        try:
            self.model = LightOnOcrForConditionalGeneration.from_pretrained(
                self.model_path, 
                torch_dtype=self.dtype
            ).to(self.device)
            self.processor = LightOnOcrProcessor.from_pretrained(
                self.model_path, 
                fix_mistral_regex=True
            )
        except Exception as e:
            print(f"Error loading OCR: {e}")
            raise e

    def run_ocr(self, image: Image.Image) -> str:
        return self.run_ocr_batch([image])[0]

    def run_ocr_batch(self, images: list[Image.Image]) -> list[str]:
        self.load()
        
        processed_images = []
        for img in images:
            # Resize for optimal OCR performance
            max_size = 1540
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            processed_images.append(img)

        prompt_text = "Analyze this medical document. Extract all visible text accurately."
        
        # Prepare batch conversations
        conversations = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            for _ in processed_images
        ]
        
        prompts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]
        
        inputs = self.processor(text=prompts, images=processed_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device=self.device, dtype=self.dtype) if torch.is_tensor(v) and v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)
        
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

class GLMOCR(OCRProvider):
    @property
    def name(self):
        return "GLM"

    def __init__(self, model_path="./GLM-OCR"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if self.model is not None:
            return
        
        print(f"Loading GLM-OCR from {self.model_path}...")
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception as e:
            print(f"Error loading GLM-OCR: {e}")
            raise e

    def run_ocr(self, image: Image.Image) -> str:
        return self.run_ocr_batch([image])[0]

    def run_ocr_batch(self, images: list[Image.Image]) -> list[str]:
        self.load()
        
        processed_images = []
        for img in images:
            max_size = 1540
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            processed_images.append(img)

        # Prepare batch content for GLM-OCR
        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img}, 
                        {"type": "text", "text": "Text Recognition:"}
                    ],
                }
            ] for img in processed_images
        ]
        
        prompts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in batch_messages
        ]
        
        inputs = self.processor(text=prompts, images=processed_images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

# ======================
# Helper Functions for Prompts
# ======================

def get_structure_prompt(ocr_text: str, pdl_context: str) -> str:
    clinical_context = """
    CONTEXT: This is a stroke follow-up form. Patients fill this out periodically.
    The goal is to track their recovery, mobility, self-care, and health status over time.
    """
    
    pdl_instruction = ""
    if pdl_context:
        pdl_instruction = f"""
        MAPPING INSTRUCTIONS:
        Use the provided Primary Data List (PDL) as a master reference.
        - For every data point found in OCR text, find the corresponding 'ref' (Reference Number) in the PDL.
        - The JSON keys MUST be the Reference Numbers (e.g., "2.000", "6.150").
        - If a field has 'normalization' codes (e.g., "1 = Male", "ACT = Australian Capital Territory"), 
          convert the extracted text/selection to the normalized Export Code (e.g., "1", "ACT").
        - If a field is 'Free text', keep the original text but ensure it's clean.
        - STRICT RULE: You MUST return EVERY SINGLE reference number from the PDL in your final JSON output, without exception.
        - If the OCR text does NOT contain ANY evidence or data for a reference number (i.e. the OCR missed it entirely), set the value to EXACTLY the string "NOT_EXTRACTED".
        - If the OCR text shows the field exists but it was left intentionally blank by the patient, set the value to null.
        - Evaluate the overall extraction quality and provide an integer for "processing_confidence" (0-100) and a string for "extraction_summary_comment".
          The "processing_confidence" should be a balance of:
          1. Data Completeness: How much of the form data was picked up by the OCR? (Lower the score if expected sections of the form are completely missing from the OCR).
          2. Data Accuracy: How clear, legible, and unambiguous is the data that WAS picked up? (Lower the score for garbled text, bad handwriting, or ambiguous mappings).
          3. Intentional Blanks: Do NOT penalize the score for fields intentionally left blank by the patient (mapped to null). A perfectly clean OCR extraction of a mostly blank form should still have a very high confidence score. You must differentiate between "OCR failed to capture this area (NOT_EXTRACTED)" vs "Patient left it blank (null)".
        - The "extraction_summary_comment" MUST be a concise one-line summary. It MUST explicitly state if any fields were completely missed by OCR (NOT_EXTRACTED) or were mapped with high uncertainty.
        
        PDL REFERENCE:
        {pdl_context}
        """
    
    return f"""
    {clinical_context}
    You are an expert medical data analyst. 
    Transform the following raw OCR text into a structured JSON object.
    {pdl_instruction}
    OCR Text:
    ---
    {ocr_text}
    ---
    Return ONLY a flat JSON object where keys are Reference Numbers from the PDL.
    """

# ======================
# LLM Implementations
# ======================

class GeminiLLM(LLMProvider):
    @property
    def name(self):
        return "GEMINI"

    def __init__(self, api_key: str, model_name="gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

    def structure_data(self, ocr_text: str, pdl_context: str, callback=None) -> dict:
        if callback: callback("Structuring with Gemini...")
        
        prompt = get_structure_prompt(ocr_text, pdl_context)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model_name, 
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text)
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(10 * (2 ** attempt))
                    continue
                raise e

class GLMLLM(LLMProvider):
    @property
    def name(self):
        return "GLM"

    def __init__(self, api_key: str, model_name="glm-4.7"):
        self.api_key = api_key
        self.model_name = model_name

    def structure_data(self, ocr_text: str, pdl_context: str, callback=None) -> dict:
        if callback: callback("Structuring with GLM (JSON Mode enabled)...")
        
        if not self.api_key:
            raise ValueError("GLM_API_KEY not found in environment.")

        client = ZaiClient(api_key=self.api_key)
        
        system_instruction = "You are an expert medical data analyst. You MUST return ONLY a raw JSON object. No markdown, no explanations. Ensure all keys match the PDL Reference numbers provided."
        user_prompt = get_structure_prompt(ocr_text, pdl_context)
        
        text_content = ""
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                thinking={"type": "disabled"},
                response_format={"type": "json_object"},
                max_tokens=4096,
                temperature=0.1
            )
            
            text_content = response.choices[0].message.content
            
            # Robust JSON extraction just in case
            if "```json" in text_content:
                text_content = text_content.split("```json")[1].split("```")[0].strip()
            elif "```" in text_content:
                text_content = text_content.split("```")[1].split("```")[0].strip()
            
            return json.loads(text_content)
        except json.JSONDecodeError as je:
            with open("glm_debug_response.txt", "w", encoding="utf-8") as f:
                # Log everything for deep debugging
                f.write(f"JSON PARSING ERROR: {str(je)}\n")
                f.write(f"RAW CONTENT: '{text_content}'\n")
                try:
                    msg_obj = response.choices[0].message
                    f.write(f"MESSAGE OBJECT: {str(msg_obj)}\n")
                    if hasattr(msg_obj, 'reasoning_content'):
                        f.write(f"REASONING CONTENT: {msg_obj.reasoning_content}\n")
                except:
                    f.write("Could not retrieve message object details.\n")
            if callback: callback("GLM JSON Parsing Error. Debug details saved to glm_debug_response.txt")
            raise je
        except Exception as e:
            if callback: callback(f"GLM Error: {e}")
            raise e

# ======================
# Factory and Global Setup
# ======================

def get_llm_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider_name == "glm":
        api_key = os.getenv("GLM_API_KEY")
        return GLMLLM(api_key)
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        # Per user request, keeping 'gemini-2.5-flash'
        return GeminiLLM(api_key, model_name="gemini-2.5-flash")

def get_ocr_provider() -> OCRProvider:
    provider_name = os.getenv("OCR_PROVIDER", "lighton").lower()
    if provider_name == "glm":
        return GLMOCR()
    else:
        return LightOnOCR()

# Global instances for singleton-like usage
ocr_engine = get_ocr_provider()
llm_engine = get_llm_provider()

# ======================
# Helper Functions
# ======================

def get_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def load_pdl(csv_path):
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        pdl_context = []
        for _, row in df.iterrows():
            ref_num = str(row.get('NEW REF NUMB', ''))
            element = str(row.get('DATA ELEMENT', ''))
            codes = str(row.get('IMPORT/EXPORT CODES', ''))
            if ref_num and ref_num != 'nan' and element and element != 'nan':
                pdl_context.append({"ref": ref_num, "element": element, "normalization": codes})
        return json.dumps(pdl_context, indent=2)
    except Exception as e:
        print(f"Error loading PDL: {e}")
        return None

# ======================
# Main Processing Pipeline
# ======================

def process_document(input_path, pdl_path="PACE/Primary Data List.csv", progress_callback=None):
    start_time = time.time()
    def log(msg, progress=None):
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        print(formatted_msg)
        if progress_callback: 
            try:
                progress_callback(msg=formatted_msg, progress=progress)
            except TypeError:
                progress_callback(formatted_msg)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: {input_path} not found.")

    log("="*30)
    log(f"PHASE 1: Starting Document Processing")
    log(f"Input: {input_path}")
    log(f"OCR Provider: {ocr_engine.name.upper()}")
    log(f"LLM Provider: {llm_engine.name.upper()}")
    log("="*30)
    all_text = ""

    if input_path.lower().endswith(".pdf"):
        log("PDF detected. Converting pages to images...", progress=5)
        images = get_images_from_pdf(input_path)
        log(f"Total pages to process: {len(images)}", progress=10)
        
        # Optimization: Use batched inference (Batch size = 2 for 8GB VRAM safety)
        batch_size = 2
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            current_batch_num = (i // batch_size) + 1
            total_batches = (len(images) + batch_size - 1) // batch_size
            
            log(f"--- Processing Batch {current_batch_num}/{total_batches} ({len(batch)} pages) ---", progress=10 + int(60 * (i / len(images))))
            
            batch_results = ocr_engine.run_ocr_batch(batch)
            
            for j, page_text in enumerate(batch_results):
                page_num = i + j + 1
                all_text += f"\n--- PAGE {page_num} ---\n{page_text}\n"
                log(f"Page {page_num} OCR complete.")
        log("All PDF pages OCR complete.", progress=75)
    else:
        log("Image detected. Starting OCR...", progress=10)
        all_text = ocr_engine.run_ocr(Image.open(input_path))
        log("Image OCR complete.", progress=75)

    # Ensure extractions directory exists
    extractions_dir = "extractions"
    os.makedirs(extractions_dir, exist_ok=True)

    log("OCR Complete. Saving raw text...", progress=80)
    ocr_filename = os.path.join(extractions_dir, f"ocr_raw_{ocr_engine.name.lower()}.txt")
    with open(ocr_filename, "w", encoding="utf-8") as f:
        f.write(all_text)
    log(f"Raw OCR text saved to {ocr_filename}")

    log("Loading Primary Data List (PDL) for mapping...")
    pdl_context = load_pdl(pdl_path)
    if pdl_context:
        log("PDL loaded successfully.")
    else:
        log("Warning: PDL context is empty or file missing.")

    try:
        log(f"LLM Provider: {llm_engine.name}", progress=85)
        def llm_callback(msg):
            log(msg, progress=90)
        result = llm_engine.structure_data(all_text, pdl_context, callback=llm_callback)
        
        result_filename = os.path.join(extractions_dir, f"result_OCR-{ocr_engine.name.upper()}_LLM-{llm_engine.name.upper()}.json")
        log(f"Saving results to {result_filename}...", progress=95)
        with open(result_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        duration = time.time() - start_time
        mins, secs = divmod(duration, 60)
        
        log("="*30)
        log("SUCCESS: Document processing complete.", progress=100)
        log(f"TOTAL DURATION: {int(mins)}m {int(secs)}s")
        log("="*30)
        return result
    except Exception as e:
        duration = time.time() - start_time
        mins, secs = divmod(duration, 60)
        log(f"LLM Error after {int(mins)}m {int(secs)}s: {e}")
        return {"error": str(e), "raw_text": all_text}

def run_ocr(image):
    """Legacy wrapper for backward compatibility with app.py."""
    return ocr_engine.run_ocr(image)

if __name__ == "__main__":
    # Test run
    # os.environ["LLM_PROVIDER"] = "gemini" 
    result = process_document(r"PACE/test_data/Test3.pdf", r"PACE/Primary Data List.csv")
    print("Execution complete. Check output files for results.")
