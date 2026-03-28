import requests
import time
from textblob import TextBlob
import folium
import csv
import html

# ========= CONFIG ==========
API_KEY = ""  # ←  Flickr API key

OUTPUT_HTML = "venice_flickr_mobility_sentiment_map_clickable.html"
CSV_OUTPUT = "venice_sentiment_coordinates.csv"

REQUEST_DELAY = 1
MAX_PHOTOS_PER_KEYWORD = 500

# Flickr BBOX format = min_lon, min_lat, max_lon, max_lat
VENICE_BBOX = "12.3000,45.4170,12.3575,45.4540"

SEARCH_KEYWORDS = [
    "boat", "walking", "wheelchair", "mobility", "transport", "gondola", "vaporetto",
    "barca", "camminare", "sedia a rotelle", "accessibilità", "mobilità"
]


# ========== Flickr Photo Search ==========
def search(keyword):
    url = "https://api.flickr.com/services/rest/"
    photos = []
    page = 1

    while len(photos) < MAX_PHOTOS_PER_KEYWORD:
        params = {
            "method": "flickr.photos.search",
            "api_key": API_KEY,
            "bbox": VENICE_BBOX,
            "text": keyword,
            "format": "json",
            "nojsoncallback": 1,
            "extras": "geo,tags,description,url_m",
            "per_page": 200,
            "page": page,
            "content_type": 1,
            "safe_search": 1,
            "has_geo": 1
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[ERROR] keyword='{keyword}', page={page}, error={e}")
            break

        result = data.get("photos", {}).get("photo", [])
        if not result:
            break

        for p in result:
            try:
                lat = float(p.get("latitude", 0))
                lon = float(p.get("longitude", 0))
            except (ValueError, TypeError):
                continue

            if lat != 0 and lon != 0:
                photos.append(p)
                if len(photos) >= MAX_PHOTOS_PER_KEYWORD:
                    break

        page += 1
        time.sleep(REQUEST_DELAY)

    print(f"[INFO] '{keyword}' => {len(photos)} results")
    return photos


# ========== Sentiment ==========
def sentiment(text):
    if not text or not text.strip():
        return 0
    return TextBlob(text).sentiment.polarity


# ========== Build Folium Map ==========
def build_map(records):
    m = folium.Map(
        location=[45.437, 12.335],
        zoom_start=14,
        tiles="https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
        attr="CartoDB Dark Matter No Labels"
    )

    # Hide zoom controls, layer controls, and copyright text
    m.get_root().html.add_child(folium.Element("""
    <style>
    .leaflet-control-attribution,
    .leaflet-control-layers,
    .leaflet-control-zoom {
        display: none !important;
    }
    </style>
    """))

    for p in records:
        pol = p["polarity"]
        color = "green" if pol > 0.1 else "red" if pol < -0.1 else "blue"

        title = html.escape(str(p.get("title", "N/A") or "N/A"))
        owner = html.escape(str(p.get("owner", "N/A") or "N/A"))
        url = p.get("url", "")
        lat = p["lat"]
        lon = p["lon"]

        if url:
            safe_url = html.escape(url, quote=True)
            url_html = f'<a href="{safe_url}" target="_blank">View Photo</a>'
        else:
            url_html = "No URL"

        popup_html = f"""
        <div style="width:260px; font-size:13px; line-height:1.5;">
            <b>Title:</b> {title}<br>
            <b>Owner:</b> {owner}<br>
            <b>Latitude:</b> {lat:.6f}<br>
            <b>Longitude:</b> {lon:.6f}<br>
            <b>Polarity:</b> {pol:.3f}<br>
            <b>URL:</b> {url_html}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    m.save(OUTPUT_HTML)
    print(f"[HTML SAVED] Map saved as {OUTPUT_HTML}")


# ========== Main ==========
def main():
    if not API_KEY or API_KEY == "YOUR_FLICKR_API_KEY":
        raise ValueError("⚠ 请先在代码里填入有效的 Flickr API_KEY")

    all_photos = {}

    # Step 1: Search Flickr
    for kw in SEARCH_KEYWORDS:
        results = search(kw)
        for p in results:
            pid = p.get("id")
            if not pid:
                continue

            if pid not in all_photos:
                txt = " ".join([
                    p.get("title", ""),
                    p.get("description", {}).get("_content", ""),
                    p.get("tags", "")
                ])

                try:
                    lat = float(p.get("latitude", 0))
                    lon = float(p.get("longitude", 0))
                except (ValueError, TypeError):
                    continue

                all_photos[pid] = {
                    "id": pid,
                    "owner": p.get("owner", ""),
                    "title": p.get("title", ""),
                    "lat": lat,
                    "lon": lon,
                    "url": p.get("url_m", ""),
                    "polarity": sentiment(txt)
                }

    print(f"Total photos collected: {len(all_photos)}")

    # Step 2: Save CSV
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Latitude", "Longitude", "Polarity", "Title", "Owner", "URL"])
        for p in all_photos.values():
            writer.writerow([
                p["lat"],
                p["lon"],
                p["polarity"],
                p["title"],
                p["owner"],
                p["url"]
            ])

    print(f"[CSV SAVED] CSV exported → {CSV_OUTPUT}")

    # Step 3: Build map
    build_map(list(all_photos.values()))


if __name__ == "__main__":
    main()
