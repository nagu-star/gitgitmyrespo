import urllib.request
import ssl
import json
import os
import time

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/metadata', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

base_url = 'https://mplads.mospi.gov.in'

def post_json(path, payload):
    url = base_url + path
    data_bytes = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
            body = resp.read().decode('utf-8', errors='ignore')
            return json.loads(body)
    except Exception as e:
        print(f"Error calling {path} with {payload}: {e}")
        return None

print("1. Fetching States...")
states = post_json('/rest/PreLoginDashboardData/getStateData', {})
if states:
    with open('data/raw/states.json', 'w', encoding='utf-8') as f:
        json.dump(states, f, indent=2)
    print(f"Saved {len(states)} states to data/raw/states.json")

print("2. Fetching Tenures...")
tenures = post_json('/rest/PreLoginDashboardData/getTenureData', {'uname': '0,0,0,2'})
if tenures:
    with open('data/raw/tenures.json', 'w', encoding='utf-8') as f:
        json.dump(tenures, f, indent=2)
    print(f"Saved {len(tenures)} tenures to data/raw/tenures.json")

print("3. Fetching MPs and Constituencies...")
combo = post_json('/rest/PreLoginDashboardData/getMpAndConstCombo', {'const_combo': '0'})
if combo:
    with open('data/raw/mp_constituency_combo.json', 'w', encoding='utf-8') as f:
        json.dump(combo, f, indent=2)
    print(f"Saved {len(combo)} MP/Constituency records to data/raw/mp_constituency_combo.json")

print("4. Fetching National Overview Tiles (17th & 18th Lok Sabha)...")
national_tiles = {}
for tenure_id, name in [("1", "17th Lok Sabha"), ("2", "18th Lok Sabha")]:
    res = post_json('/rest/PreLoginDashboardData/getTilesData', {'uname': f"0,0,0,{tenure_id}"})
    if res:
        national_tiles[name] = res

with open('data/raw/national_tiles.json', 'w', encoding='utf-8') as f:
    json.dump(national_tiles, f, indent=2)
print("Saved national tiles data.")

print("5. Fetching State-wise Tiles for 18th & 17th Lok Sabha...")
state_tiles = []
if states:
    for s in states:
        s_id = s.get('STATE_ID')
        s_name = s.get('STATE_NAME')
        print(f"Fetching tiles for State: {s_name} (ID: {s_id})...")
        for t_id, t_name in [("2", "18th Lok Sabha"), ("1", "17th Lok Sabha")]:
            res = post_json('/rest/PreLoginDashboardData/getTilesData', {'uname': f"{s_id},0,0,{t_id}"})
            if res and isinstance(res, dict):
                record = {
                    "STATE_ID": s_id,
                    "STATE_NAME": s_name,
                    "TENURE_ID": t_id,
                    "TENURE_NAME": t_name,
                    "metrics": res
                }
                state_tiles.append(record)
            time.sleep(0.1)

with open('data/raw/state_tiles.json', 'w', encoding='utf-8') as f:
    json.dump(state_tiles, f, indent=2)
print(f"Saved {len(state_tiles)} state tile records to data/raw/state_tiles.json")

print("6. Fetching State-wise Constituencies...")
constituencies_by_state = []
if states:
    for s in states:
        s_id = s.get('STATE_ID')
        s_name = s.get('STATE_NAME')
        c_list = post_json('/rest/PreLoginDashboardData/getConstituencyData', {'id': str(s_id)})
        if c_list and isinstance(c_list, list):
            for c in c_list:
                constituencies_by_state.append({
                    "STATE_ID": s_id,
                    "STATE_NAME": s_name,
                    "CONSTITUENCY_ID": c.get('ID'),
                    "CONSTITUENCY_NAME": c.get('CAPTION')
                })
        time.sleep(0.05)

with open('data/raw/constituencies.json', 'w', encoding='utf-8') as f:
    json.dump(constituencies_by_state, f, indent=2)
print(f"Saved {len(constituencies_by_state)} constituencies to data/raw/constituencies.json")

print("7. Fetching Reviews...")
reviews = post_json('/rest/PreLoginCitizenWorkRcmdRest/getReviewDetailsByWork', {})
if reviews:
    with open('data/raw/work_reviews.json', 'w', encoding='utf-8') as f:
        json.dump(reviews, f, indent=2)
    print(f"Saved {len(reviews)} reviews to data/raw/work_reviews.json")

print("\nData Extraction complete!")
