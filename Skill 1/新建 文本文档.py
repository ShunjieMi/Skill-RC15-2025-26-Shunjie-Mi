!pip -q install folium branca shapely pyproj

import io, zipfile, re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from shapely.geometry import LineString, Point
from pyproj import Transformer
import folium
from folium import FeatureGroup
from folium.raster_layers import TileLayer
from branca.colormap import LinearColormap
from google.colab import files


# =========================
# 可调参数
# =========================
RESAMPLE_STEP_M = 2.0     # 路线重采样间隔(米)
SMOOTH_WINDOW_M = 30.0    # 平滑窗口(米)

# 线更细一点（你要再细：CORE_WEIGHT=2, GLOW_WEIGHT=4）
GLOW_WEIGHT = 6
GLOW_OPACITY = 0.18
CORE_WEIGHT = 3
CORE_OPACITY = 0.95

# 点
POINT_RADIUS = 1
POINT_OPACITY = 0.95      # 点更亮一点，像你图里的“星点”
POINT_BORDER = 0          # 0 无边框

# 默认显示
SHOW_BASEMAP = True
SHOW_BLANK = False
SHOW_ROUTE_BASE = False

SHOW_GSR_LINE = True
SHOW_BPM_LINE = False
SHOW_FLEX_LINE = False

SHOW_GSR_POINTS = True
SHOW_BPM_POINTS = False
SHOW_FLEX_POINTS = False


# =========================
# 1) 上传 CSV + KMZ/KML
# =========================
uploaded = files.upload()
csv_name, route_name = None, None
for n in uploaded:
    if n.lower().endswith(".csv"):
        csv_name = n
    if n.lower().endswith(".kmz") or n.lower().endswith(".kml"):
        route_name = n

if not csv_name or not route_name:
    raise RuntimeError("请同时上传：CSV 点数据 + KMZ/KML 路线文件")

print("✅ CSV  :", csv_name)
print("✅ Route:", route_name)


# =========================
# 2) 读取 CSV（自动识别列）
# =========================
raw_csv = uploaded[csv_name]
try:
    df = pd.read_csv(io.BytesIO(raw_csv), encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(io.BytesIO(raw_csv), encoding="gbk")

def pick_col(cols, candidates):
    m = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in m:
            return m[cand]
    return None

cols = df.columns.tolist()
time_col = pick_col(cols, ["time", "timestamp", "datetime", "date_time", "date"])
lat_col  = pick_col(cols, ["lat", "latitude"])
lon_col  = pick_col(cols, ["lon", "lng", "long", "longitude"])
gsr_col  = pick_col(cols, ["gsr", "eda"])
bpm_col  = pick_col(cols, ["bpm", "hr", "heart_rate", "heartrate", "heart rate"])
flex_col = pick_col(cols, ["flex"])

needed = [lat_col, lon_col, gsr_col, bpm_col, flex_col]
if any(x is None for x in needed):
    raise ValueError(
        "❌ 没识别到必要列。请确认 CSV 有：lat/lng + gsr + bpm/hr(heart_rate) + flex。\n"
        f"当前列名：{cols}"
    )

rename_map = {lat_col:"lat", lon_col:"lon", gsr_col:"gsr", bpm_col:"bpm", flex_col:"flex"}
if time_col is not None:
    rename_map[time_col] = "time"
df = df.rename(columns=rename_map).copy()

for c in ["lat","lon","gsr","bpm","flex"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
else:
    df["time"] = np.arange(len(df))

df = df.dropna(subset=["lat","lon","gsr","bpm","flex"]).copy()
df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)].copy()
df = df.sort_values("time").reset_index(drop=True)

print("✅ 清洗后点数:", len(df))


# =========================
# 3) 解析 KMZ/KML 路线坐标 (lon,lat)
# =========================
def extract_kml_text(route_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".kml"):
        return route_bytes.decode("utf-8", errors="ignore")
    zf = zipfile.ZipFile(io.BytesIO(route_bytes))
    kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
    if not kml_files:
        raise ValueError("KMZ里没有找到KML文件")
    return zf.read(kml_files[0]).decode("utf-8", errors="ignore")

def parse_route_lonlat(kml_text: str):
    candidates = []

    # <coordinates>
    try:
        root = ET.fromstring(kml_text)
        for elem in root.iter():
            tag = elem.tag.lower()
            if tag.endswith("coordinates") and elem.text:
                text = elem.text.strip()
                parts = re.split(r"\s+", text)
                pts = []
                for p in parts:
                    if not p:
                        continue
                    vals = p.split(",")
                    if len(vals) >= 2:
                        try:
                            lon = float(vals[0]); lat = float(vals[1])
                            pts.append((lon, lat))
                        except:
                            pass
                if len(pts) >= 2:
                    candidates.append(pts)
    except Exception:
        pass

    # gx:Track
    gx = re.findall(r"<gx:coord>\s*([-\d\.]+)\s+([-\d\.]+)(?:\s+[-\d\.]+)?\s*</gx:coord>", kml_text)
    if gx:
        pts = [(float(lon), float(lat)) for lon, lat in gx]
        if len(pts) >= 2:
            candidates.append(pts)

    if not candidates:
        raise ValueError("❌ 没有从KML解析到路线")

    candidates.sort(key=len, reverse=True)
    return candidates[0]

def clean_route_lonlat(route_lonlat):
    cleaned = [(float(lon), float(lat)) for lon, lat in route_lonlat
               if np.isfinite(lon) and np.isfinite(lat)]
    dedup = []
    for pt in cleaned:
        if not dedup or pt != dedup[-1]:
            dedup.append(pt)
    if len(dedup) < 2:
        raise ValueError("❌ 路线有效点少于2个")
    return dedup

kml_text = extract_kml_text(uploaded[route_name], route_name)
route_lonlat = clean_route_lonlat(parse_route_lonlat(kml_text))
print("✅ 路线有效点数:", len(route_lonlat))


# =========================
# 4) 投影到路线（UTM）
# =========================
lons = np.array([p[0] for p in route_lonlat], dtype=float)
lats = np.array([p[1] for p in route_lonlat], dtype=float)
lon0 = float(np.median(lons))
lat0 = float(np.median(lats))
zone = int((lon0 + 180) // 6) + 1
epsg_utm = 32600 + zone if lat0 >= 0 else 32700 + zone

to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
to_wgs = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)

route_xy = [to_utm.transform(lon, lat) for lon, lat in route_lonlat]
route_xy2 = []
for pt in route_xy:
    if not route_xy2 or pt != route_xy2[-1]:
        route_xy2.append(pt)

line = LineString(route_xy2)
if line.is_empty or (not np.isfinite(line.length)) or line.length <= 0:
    raise ValueError("❌ 路线 LineString 无效")

snapped_lats, snapped_lons, dist_along_m = [], [], []
for r in df.itertuples(index=False):
    x, y = to_utm.transform(float(r.lon), float(r.lat))
    dproj = line.project(Point(x, y))
    p = line.interpolate(dproj)
    lon_s, lat_s = to_wgs.transform(p.x, p.y)
    snapped_lons.append(lon_s)
    snapped_lats.append(lat_s)
    dist_along_m.append(float(dproj))

df["lon_snap"] = snapped_lons
df["lat_snap"] = snapped_lats
df["route_m"] = dist_along_m


# =========================
# 5) 平滑 + 等距重采样（线用）
# =========================
d = df.sort_values("route_m").reset_index(drop=True)

spacing = np.median(np.diff(d["route_m"].values)) if len(d) > 2 else 1.0
if not np.isfinite(spacing) or spacing <= 0:
    spacing = 1.0
window_pts = max(3, int(round(SMOOTH_WINDOW_M / spacing)))
if window_pts % 2 == 0:
    window_pts += 1

for col in ["gsr", "bpm", "flex"]:
    d[col + "_smooth"] = (
        d[col].rolling(window=window_pts, center=True, min_periods=max(2, window_pts//3))
        .mean()
        .interpolate(limit_direction="both")
    )

m_min, m_max = float(d["route_m"].min()), float(d["route_m"].max())
grid_m = np.arange(m_min, m_max + RESAMPLE_STEP_M, RESAMPLE_STEP_M)

def interp(x, y, x_new):
    return np.interp(x_new, x.astype(float).values, y.astype(float).values)

grid = pd.DataFrame({
    "route_m": grid_m,
    "lat":  interp(d["route_m"], d["lat_snap"], grid_m),
    "lon":  interp(d["route_m"], d["lon_snap"], grid_m),
    "gsr":  interp(d["route_m"], d["gsr_smooth"],  grid_m),
    "bpm":  interp(d["route_m"], d["bpm_smooth"],  grid_m),
    "flex": interp(d["route_m"], d["flex_smooth"], grid_m),
})

print(f"✅ 平滑窗口: {window_pts} 点 (~{SMOOTH_WINDOW_M}m)，重采样点数: {len(grid)}")


# =========================
# 6) 三套渐变色（线和点必须同色）
#     注意：点用“原始 df 的值范围”来缩放，更符合你点的真实分布
# =========================
cm_gsr_base  = LinearColormap(["#2b0b3f", "#6a1b9a", "#b23aee", "#ff4fd8", "#ffd1ff"], caption="GSR")
cm_bpm_base  = LinearColormap(["#001a4d", "#0047ff", "#00b3ff", "#00ffd5", "#e8ffff"], caption="Heart Rate (BPM)")
cm_flex_base = LinearColormap(["#3b1d00", "#ff6a00", "#ffb000", "#ffe66d", "#fff7cc"], caption="Flex")

def scale_cm(cm, series):
    vmin, vmax = np.nanpercentile(series.astype(float).values, [2, 98])
    return cm.scale(float(vmin), float(vmax)), float(vmin), float(vmax)

cm_gsr,  gmin, gmax = scale_cm(cm_gsr_base,  df["gsr"])
cm_bpm,  bmin, bmax = scale_cm(cm_bpm_base,  df["bpm"])
cm_flex, fmin, fmax = scale_cm(cm_flex_base, df["flex"])

def color_of(v, cm, vmin, vmax):
    vv = float(np.clip(v, vmin, vmax))
    return cm(vv)


# =========================
# 7) 画图：底图开关 + 黑底开关 + 线开关 + 3个点开关(按值上色)
# =========================
center = [float(np.median(grid["lat"])), float(np.median(grid["lon"]))]

m = folium.Map(location=center, zoom_start=15, tiles=None)

# Basemap（干净无文字）
carto_nolabels = "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
carto_attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors ' \
             '&copy; <a href="https://carto.com/attributions">CARTO</a>'
TileLayer(tiles=carto_nolabels, attr=carto_attr, name="Basemap", overlay=False, control=True, show=SHOW_BASEMAP).add_to(m)

# 纯黑底（作为替代底图）
black_png_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO1Hk9kAAAAASUVORK5CYII="
TileLayer(tiles=black_png_data_url, attr="blank", name="Blank (Black)", overlay=False, control=True, show=SHOW_BLANK).add_to(m)

# Route base（可选）
route_base = FeatureGroup(name="Route (Base)", show=SHOW_ROUTE_BASE)
route_latlng = [(lat, lon) for lon, lat in route_lonlat]
folium.PolyLine(route_latlng, color="#777777", weight=2, opacity=0.55).add_to(route_base)
route_base.add_to(m)

# 渐变线（3条）
def add_glow_line_layer(name, col, cm, vmin, vmax, show=False):
    fg = FeatureGroup(name=name, show=show)
    for i in range(len(grid) - 1):
        v = float(grid.loc[i, col])
        c = color_of(v, cm, vmin, vmax)
        p1 = (float(grid.loc[i, "lat"]), float(grid.loc[i, "lon"]))
        p2 = (float(grid.loc[i+1, "lat"]), float(grid.loc[i+1, "lon"]))
        folium.PolyLine([p1, p2], color=c, weight=GLOW_WEIGHT, opacity=GLOW_OPACITY).add_to(fg)
        folium.PolyLine([p1, p2], color=c, weight=CORE_WEIGHT, opacity=CORE_OPACITY).add_to(fg)
    fg.add_to(m)

add_glow_line_layer("GSR Glow", "gsr", cm_gsr, gmin, gmax, show=SHOW_GSR_LINE)
add_glow_line_layer("Heart Rate Glow", "bpm", cm_bpm, bmin, bmax, show=SHOW_BPM_LINE)
add_glow_line_layer("Flex Glow", "flex", cm_flex, fmin, fmax, show=SHOW_FLEX_LINE)

# 点图层：3个开关，按值映射同色带
def add_points_layer(layer_name, value_col, cm, vmin, vmax, show=False):
    fg = FeatureGroup(name=layer_name, show=show)
    for r in df.itertuples(index=False):
        v = float(getattr(r, value_col))
        c = color_of(v, cm, vmin, vmax)
        folium.CircleMarker(
            location=(float(r.lat_snap), float(r.lon_snap)),
            radius=POINT_RADIUS,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=POINT_OPACITY,
            opacity=POINT_OPACITY,
            weight=POINT_BORDER,
            tooltip=f"{layer_name}: {v:.3f}"
        ).add_to(fg)
    fg.add_to(m)

add_points_layer("GSR Points", "gsr", cm_gsr, gmin, gmax, show=SHOW_GSR_POINTS)
add_points_layer("Heart Rate Points", "bpm", cm_bpm, bmin, bmax, show=SHOW_BPM_POINTS)
add_points_layer("Flex Points", "flex", cm_flex, fmin, fmax, show=SHOW_FLEX_POINTS)

# 图例（全局显示）
m.add_child(cm_gsr)
m.add_child(cm_bpm)
m.add_child(cm_flex)

folium.LayerControl(collapsed=False).add_to(m)


# =========================
# 8) 导出
# =========================
out_csv = "snapped_to_route.csv"
out_html = "final_map_points_value_colored.html"

df.to_csv(out_csv, index=False, encoding="utf-8-sig")
m.save(out_html)

print("✅ 输出：", out_csv, out_html)
files.download(out_csv)
files.download(out_html)

m