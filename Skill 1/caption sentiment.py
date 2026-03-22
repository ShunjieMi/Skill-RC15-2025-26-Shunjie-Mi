import time
import requests
import pandas as pd
import torch

from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from textblob import TextBlob


# =========================================================
# CONFIG
# =========================================================
FLICKR_API_KEY = "88d2aed04a09d4dd50bd1f404b7fac54"
FLICKR_ENDPOINT = "https://www.flickr.com/services/rest/"

# Bounding box (min_lon, min_lat, max_lon, max_lat)
CASTELLO_BBOX = (12.3348, 45.42430, 12.36928, 45.44128)

# Recommended: start with a small sample size for testing
TARGET_N = 100

YEAR_RANGES = [
    ("2018-01-01", "2018-12-31"),
    ("2019-01-01", "2019-12-31"),
    ("2020-01-01", "2020-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

RAW_METADATA_CSV = "castello_bbox_flickr_metadata.csv"
CAPTIONED_CSV = "castello_bbox_captioned_noprompt.csv"
FILTERED_CSV = "castello_bbox_captioned_noprompt_filtered.csv"
FINAL_OUTPUT_CSV = "castello_bbox_captioned_noprompt_filtered_sentiment.csv"

# Only exclude indoor scenes (do NOT exclude Biennale)
EXCLUDE_KEYWORDS = [
    "indoor", "interior", "room", "inside", "indoors",
    "ceiling", "floor", "corridor", "hallway",
    "living room", "bedroom", "kitchen", "bathroom",
    "gallery interior", "museum interior", "indoor exhibition"
]

# Keep outdoor / mobility / urban scene related content
INCLUDE_KEYWORDS = [
    "street", "bridge", "canal", "water", "walk", "walking", "pedestrian",
    "alley", "outdoor", "city", "people", "boat", "square", "public",
    "crowd", "crowded", "walkway", "narrow", "open", "steps", "step",
    "path", "passage", "canal-side", "building"
]

KEYWORDS = [
    "bridge", "stairs", "step", "ramp", "canal", "boat", "water",
    "street", "walkway", "pedestrian", "people", "crowd", "crowded",
    "narrow", "wide", "open", "busy", "calm", "quiet", "stressful",
    "public space", "square", "access", "accessible", "mobility",
    "stroller", "cart", "wheelchair", "elderly", "luggage",
    "shopping", "delivery", "bench", "doorway", "tourist",
    "bridge crossing", "canal-side", "walking space", "alley", "building"
]


# =========================================================
# STEP 1. FLICKR SEARCH
# =========================================================
def flickr_search_bbox(api_key, bbox, min_taken_date, max_taken_date, page=1, per_page=250):
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "bbox": ",".join(map(str, bbox)),
        "has_geo": 1,
        "media": "photos",
        "content_types": "0",
        "safe_search": 1,
        "extras": "geo,date_taken,tags,owner_name,url_m,url_c,url_l",
        "format": "json",
        "nojsoncallback": 1,
        "per_page": per_page,
        "page": page,
        "sort": "date-taken-desc",
        "min_taken_date": min_taken_date,
        "max_taken_date": max_taken_date,
    }

    response = requests.get(FLICKR_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("stat") != "ok":
        raise RuntimeError(f"Flickr API error: {data}")

    return data["photos"]


def best_image_url(photo):
    return photo.get("url_l") or photo.get("url_c") or photo.get("url_m")


def collect_flickr_metadata(api_key, bbox, target_n=100):
    rows = []
    seen = set()

    for min_date, max_date in YEAR_RANGES:
        try:
            first_page = flickr_search_bbox(
                api_key=api_key,
                bbox=bbox,
                min_taken_date=min_date,
                max_taken_date=max_date,
                page=1,
                per_page=250,
            )
        except Exception as e:
            print(f"Failed year slice {min_date} to {max_date}: {e}")
            continue

        total = int(first_page.get("total", 0))
        pages = int(first_page.get("pages", 0))
        max_pages = min(pages, 16)

        print(f"{min_date} -> {max_date} | total={total} pages={pages} using={max_pages}")

        for page in range(1, max_pages + 1):
            try:
                result = flickr_search_bbox(
                    api_key=api_key,
                    bbox=bbox,
                    min_taken_date=min_date,
                    max_taken_date=max_date,
                    page=page,
                    per_page=250,
                )
            except Exception as e:
                print(f"  failed page {page}: {e}")
                continue

            photos = result.get("photo", [])
            print(f"  page={page} returned={len(photos)}")

            for p in photos:
                pid = p.get("id")
                if not pid or pid in seen:
                    continue

                lat = p.get("latitude")
                lon = p.get("longitude")
                url = best_image_url(p)

                if not lat or not lon or lat == "0" or lon == "0":
                    continue
                if not url:
                    continue

                seen.add(pid)

                rows.append({
                    "photo_id": pid,
                    "title": p.get("title", ""),
                    "tags": p.get("tags", ""),
                    "owner": p.get("ownername", ""),
                    "date_taken": p.get("datetaken", ""),
                    "lat": float(lat),
                    "lon": float(lon),
                    "image_url": url,
                    "time_slice_start": min_date,
                    "time_slice_end": max_date,
                })

                if len(rows) >= target_n:
                    return pd.DataFrame(rows)

    return pd.DataFrame(rows)


# =========================================================
# STEP 2. LOAD BLIP
# =========================================================
def load_blip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BLIP model on: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    return processor, model, device


# =========================================================
# STEP 3. NO-PROMPT CAPTIONING
# =========================================================
def caption_image_from_url(image_url, processor, model, device):
    response = requests.get(image_url, timeout=20)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=3
        )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption


def add_captions(df):
    processor, model, device = load_blip_model()

    captions = []
    failures = 0

    for idx, row in df.iterrows():
        try:
            caption = caption_image_from_url(
                image_url=row["image_url"],
                processor=processor,
                model=model,
                device=device,
            )
        except Exception as e:
            caption = ""
            failures += 1
            print(f"caption failed for {row['photo_id']}: {e}")

        captions.append(caption)

        if (idx + 1) % 10 == 0:
            print(f"captioned {idx + 1}/{len(df)} | failures={failures}")

        time.sleep(0.1)

    out = df.copy()
    out["caption"] = captions
    return out


# =========================================================
# STEP 4. FILTER
# =========================================================
def contains_any(text, keywords):
    text = str(text).lower()
    return any(k in text for k in keywords)


def filter_samples(df):
    out = df.copy()

    for col in ["title", "tags", "caption"]:
        out[col] = out[col].fillna("").astype(str).str.lower()

    out["combined_text"] = out["title"] + " " + out["tags"] + " " + out["caption"]

    # Remove indoor scenes only
    out = out[~out["combined_text"].apply(lambda x: contains_any(x, EXCLUDE_KEYWORDS))]

    # Keep outdoor / mobility / urban-related content
    out = out[out["combined_text"].apply(lambda x: contains_any(x, INCLUDE_KEYWORDS))]

    return out


# =========================================================
# STEP 5. SENTIMENT
# =========================================================
def caption_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0

    blob = TextBlob(text)
    s = blob.sentiment
    return float(s.polarity), float(s.subjectivity)


def add_sentiment(df):
    polarity_list = []
    subjectivity_list = []

    for caption in df["caption"].fillna(""):
        polarity, subjectivity = caption_sentiment(caption)
        polarity_list.append(polarity)
        subjectivity_list.append(subjectivity)

    out = df.copy()
    out["polarity"] = polarity_list
    out["subjectivity"] = subjectivity_list
    out["emotion_score"] = out["polarity"]
    return out


# =========================================================
# STEP 6. KEYWORDS
# =========================================================
def extract_keywords_from_caption(text):
    text = str(text).lower()
    found = [kw for kw in KEYWORDS if kw in text]
    return ", ".join(found)


def add_keywords(df):
    out = df.copy()
    out["keywords"] = out["caption"].fillna("").apply(extract_keywords_from_caption)
    return out