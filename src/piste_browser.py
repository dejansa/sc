import argparse
import json
import sqlite3
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = "data/piste/data.db"
ResortWays = Dict[str, Dict[str, List[Dict[str, Any]]]]


def get_resort_ways(resort_name: str) -> ResortWays:
    """Return a map of resort names to their way metadata."""
    resort_ways: ResortWays = {}
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, tag_value
        FROM relation_with_tags
        WHERE (tag_key='name' AND tag_value LIKE ?) OR (tag_key='name:en' AND tag_value LIKE ?)
        LIMIT 100
    ''', (f"%{resort_name}%", f"%{resort_name}%"))
    relations = cursor.fetchall()
    for relation in relations:
        resort_key = relation[1]
        resort_ways.setdefault(resort_key, {"ways": []})
        cursor.execute('''
            SELECT member_type, member_ref, role
            FROM relation_with_members
            WHERE member_type in ('way', 'node') AND id=?
        ''', (relation[0],))
        members = cursor.fetchall()
        for member in members:
            if member[0] == 'way':
                current_way = {"type": member[0], "nodes": []}
                resort_ways[resort_key]["ways"].append(current_way)
                cursor.execute('''
                    SELECT tag_key, tag_value
                    FROM way_with_tags
                    WHERE id=?
                ''', (member[1],))
                way_tags = cursor.fetchall()
                for way_tag in way_tags:
                    current_way[way_tag[0]] = way_tag[1]
                cursor.execute('''
                    SELECT node_id
                    FROM way_with_nodes
                    WHERE id=?
                    ORDER BY ord
                ''', (member[1],))
                node_refs = cursor.fetchall()
                for node_ref in node_refs:
                    cursor.execute('''
                        SELECT id, lat, lon
                        FROM nodes
                        WHERE id=?
                    ''', (node_ref[0],))
                    node_rows = cursor.fetchall()
                    if not node_rows:
                        continue
                    node = node_rows[0]
                    current_way["nodes"].append(
                        {"lat": float(node[1]), "lon": float(node[2])}
                    )
            elif member[0] == 'node':
                current_way = {"type": member[0], "role": member[2], "nodes": []}
                resort_ways[resort_key]["ways"].append(current_way)
                cursor.execute('''
                    SELECT id, lat, lon
                    FROM nodes
                    WHERE id=?
                ''', (member[1],))
                node_rows = cursor.fetchall()
                if not node_rows:
                    continue
                node = node_rows[0]
                current_way["nodes"].append(
                    {"lat": float(node[1]), "lon": float(node[2])}
                )
    conn.close()
    return resort_ways


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the piste browser script."""
    parser = argparse.ArgumentParser(
        description="List resort relation details from the piste database."
    )
    parser.add_argument(
        "-r",
        "--resort",
        default="Kopaonik",
        help="Resort name to query",
    )
    parser.add_argument(
        "-p",
        "--piste",
        default=None,
        help="Piste name to query",
    )
    parser.add_argument(
        "-m",
        "--map",
        action="store_true",
        help="Open a map showing nodes for the selected piste",
    )
    return parser.parse_args()


def show_resort_details(resort_ways: ResortWays) -> None:
    """Print overview details for every resort captured in the database."""
    for resort_name, resort_data in resort_ways.items():
        print(f"Resort: {resort_name}")
        for way in resort_data["ways"]:
            if "aerialway" in way or "role" in way:
                continue
            name = way.get("name")
            if not name:
                continue
            ref = way.get("ref") or way.get("piste:ref") or "??"
            difficulty = way.get("piste:difficulty", "??")
            print(f"  {ref}) {name}:  dificulty: {difficulty}")
            tags = {k: v for k, v in way.items() if k not in ["type", "nodes"]}
            print(f"    Tags: {tags}")
        print()


def show_piste_details(
    resort_ways: ResortWays, piste_name: str
) -> List[Dict[str, float]]:
    """Print selected pistes and return the node coordinates used for mapping."""
    matched_nodes: List[List[Dict[str, float]]] = []
    query_raw = piste_name.strip()
    query = query_raw.lower()
    is_ref_query = query_raw.isdigit()
    for resort_name, resort_data in resort_ways.items():
        print(f"Resort: {resort_name}")
        for way in resort_data["ways"]:
            if "aerialway" in way or "role" in way:
                continue
            name = way.get("name", "")
            ref_value = way.get("ref") or way.get("piste:ref") or ""
            if is_ref_query:
                if ref_value != query_raw:
                    continue
            else:
                if query and query not in name.lower():
                    continue
            ref_display = ref_value or "??"
            difficulty = way.get("piste:difficulty", "??")
            print(f"  {ref_display}) {name}:  dificulty: {difficulty}")
            tags = {k: v for k, v in way.items() if k not in ["type", "nodes"]}
            print(f"    Tags: {tags}")
            way_nodes = []
            for node in way["nodes"]:
                lat = float(node["lat"])
                lon = float(node["lon"])
                print(f"    Node: lat={lat}, lon={lon}")
                way_nodes.append({"lat": lat, "lon": lon})
            if way_nodes:
                matched_nodes.append(way_nodes)
        print()
    return matched_nodes


def create_piste_map(
    nodes: List[Dict[str, float]], resort_name: str, piste_name: str
) -> Path | None:
    """Render a small Leaflet map for the nodes and open it in the browser."""
    if not nodes or not any(nodes):
        print("No nodes were collected for the requested piste; skipping map.")
        return None
    safe_name = "".join(
        c if c.isalnum() else "_" for c in f"{resort_name}_{piste_name}"
    )
    # nodes is now a list of lists (ways), each way is a list of dicts with lat/lon
    ways_points = [[ [float(node["lat"]), float(node["lon"])] for node in way ] for way in nodes]
    ways_points_js = json.dumps(ways_points)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>{resort_name} - {piste_name} map</title>
<link
    rel=\"stylesheet\"
    href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"
/>
<style>
    body,
    html {{
        margin: 0;
        height: 100%;
    }}
    #map {{
        width: 100%;
        height: 100vh;
    }}
 </style>
 </head>
 <body>
 <div id=\"map\"></div>
 <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
 <script>
     const ways_points = {ways_points_js};
     if (!window.L) {{
         document.getElementById('map').innerHTML =
             'Leaflet failed to load. Check network access to unpkg.com.';
     }} else {{
         const map = L.map('map');
         L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
             maxZoom: 19,
             attribution: '&copy; OpenStreetMap contributors',
         }}).addTo(map);

        // Draw a polyline for each way
        let allPoints = [];
        for (const points of ways_points) {{
            if (points.length > 1) {{
                L.polyline(points, {{ color: 'blue', weight: 3 }}).addTo(map);
            }}
            // Draw circle markers for each point
            for (const point of points) {{
                L.circleMarker(point, {{ radius: 5 }}).addTo(map);
                allPoints.push(point);
            }}
        }}

        if (allPoints.length > 0) {{
            const bounds = L.latLngBounds(allPoints);
            map.fitBounds(bounds.pad(0.15));
        }}
     }}
 </script>
 </body>
 </html>"""
    # Use a unique filename per run to avoid browser caching a stale local file.
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"piste_map_{safe_name}_",
        suffix=".html",
        delete=False,
        encoding="utf-8",
    ) as temp_file:
        temp_file.write(html)
        map_path = Path(temp_file.name)
    print(f"Written map to {map_path}; opening in browser...")
    webbrowser.open(map_path.as_uri())
    return map_path


def main() -> None:
    """Run the primary CLI workflow."""
    args = parse_arguments()
    # print(args)
    resort_ways = get_resort_ways(args.resort)
    if args.piste:
        matched_nodes = show_piste_details(resort_ways, args.piste)
        if args.map:
            create_piste_map(matched_nodes, args.resort, args.piste)
    else:
        # print(f"resort_ways: {json.dumps(resort_ways, indent=2)}")
        print(f"resort_ways: {resort_ways}")
        show_resort_details(resort_ways)


if __name__ == "__main__":
    main()
