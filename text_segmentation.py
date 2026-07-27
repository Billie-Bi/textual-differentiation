import os
import re

import pandas as pd

INPUT_FILE = "data/To_the_Lighthouse.txt"
OUTPUT_DIR = "Textual_Indicators_output/text_segmentation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "to_the_lighthouse_paragraphs.csv")

def extract_parts(text):
    pattern = r'(THE WINDOW|TIME PASSES|THE LIGHTHOUSE)'
    chunks = re.split(pattern, text)
    
    parts = []
    for i in range(1, len(chunks), 2):
        part_name = chunks[i].title()
        part_text = chunks[i+1].strip()
        if part_text:
            parts.append((part_name, part_text))
    return parts

def extract_chapters(part_text):
    pattern = r'^\s*(\d+)\s*$'
    chunks = re.split(pattern, part_text, flags=re.MULTILINE)
    
    chapters = []
    for i in range(1, len(chunks), 2):
        chapter_id = int(chunks[i])
        chapter_text = chunks[i+1].strip()
        if chapter_text:
            chapters.append((chapter_id, chapter_text))
    return chapters

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"[ERROR] Input file missing: {INPUT_FILE}")
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
        
    data = []
    parts = extract_parts(text)
    
    for part_name, part_text in parts:
        chapters = extract_chapters(part_text)
        for chapter_id, chapter_text in chapters:
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', chapter_text) if p.strip()]
            for para_idx, paragraph_text in enumerate(paragraphs, start=1):
                data.append({
                    'part': part_name,
                    'chapter_id': chapter_id,
                    'para_idx': para_idx,
                    'text': paragraph_text
                })
                
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print(f"[INFO] Segmentation complete. Saved {len(df)} paragraphs.")
    print(f"[INFO] Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()