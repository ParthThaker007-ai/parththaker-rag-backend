import os
import sys
import json
import uuid
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import embedding, vector_store
from app.config import config
import PyPDF2

DATA_DIR = Path(__file__).parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num}]\n{page_text}\n"
                except:
                    continue
        return text
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100) -> list:
    """Sentence‑aware chunking to keep definitions and formulas intact."""
    # Split by sentences (preserve periods)
    sentences = []
    for line in text.split('\n'):
        if line.strip():
            # Split by '. ' to get sentences
            parts = line.split('. ')
            for part in parts:
                if part.strip():
                    sentences.append(part.strip() + '.')
    
    chunks = []
    current = ''
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += ' ' + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start new chunk with overlap
            overlap_text = current.split()[-overlap:] if len(current.split()) > overlap else []
            current = ' '.join(overlap_text) + ' ' + sent if overlap_text else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks

def ingest_papers():
    print("📚 Starting paper ingestion...")
    
    vector_store.ensure_collection(vector_size=384)
    
    all_papers = []
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️ No PDF files found in data/papers/")
        return
    
    print(f"📄 Found {len(pdf_files)} papers")
    
    for pdf_file in pdf_files:
        print(f"📖 Processing: {pdf_file.name}")
        
        full_text = extract_text_from_pdf(pdf_file)
        if not full_text or len(full_text) < 100:
            print(f"⚠️ Skipping {pdf_file.name} (no text extracted or too short)")
            continue
        
        chunks = chunk_text(full_text, chunk_size=400, overlap=100)
        if not chunks:
            print(f"⚠️ Skipping {pdf_file.name} (no chunks created)")
            continue
            
        print(f"   ✂️ Created {len(chunks)} chunks")
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            page = None
            for line in chunk.split("\n")[:5]:
                if "[Page" in line:
                    try:
                        page = int(line.split("[Page")[1].split("]")[0].strip())
                    except:
                        pass
                    break
            
            chunk_data.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": pdf_file.stem,
                "page": page,
                "chunk_index": i
            })
        
        all_papers.extend(chunk_data)
        
        batch_size = 50
        total_batches = (len(chunk_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(chunk_data), batch_size):
            batch = chunk_data[batch_idx:batch_idx+batch_size]
            texts = [c["text"] for c in batch]
            
            print(f"   🔄 Embedding batch {batch_idx//batch_size + 1}/{total_batches}...")
            try:
                embeddings = embedding.get_embeddings(texts)
                if embeddings and len(embeddings) == len(batch):
                    vector_store.upsert_chunks(batch, embeddings)
                    print(f"   ✅ Uploaded {len(batch)} chunks")
                else:
                    print(f"   ❌ Embedding mismatch: got {len(embeddings) if embeddings else 0}, expected {len(batch)}")
            except Exception as e:
                print(f"   ❌ Batch failed: {e}")
                continue
    
    if all_papers:
        chunks_file = CHUNKS_DIR / "all_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(all_papers, f, indent=2, ensure_ascii=False)
        print(f"💾 Chunks saved to {chunks_file}")
        print(f"✅ Ingestion complete! Total chunks: {len(all_papers)}")
    else:
        print("⚠️ No chunks were created. Check your PDF files.")

if __name__ == "__main__":
    ingest_papers()