from pathlib import Path
import requests, orjson, time
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
def fetch_nvd(start_year:int, end_year:int, out:Path):
    all_items=[]
    for year in range(start_year, end_year+1):
        params={'pubStartDate':f'{year}-01-01T00:00:00.000','pubEndDate':f'{year}-12-31T23:59:59.999','startIndex':0,'resultsPerPage':2000}
        r=requests.get(NVD_URL, params=params, timeout=30)
        r.raise_for_status()
        data=r.json()
        items=data.get('vulnerabilities',[])
        all_items.extend(items)
        time.sleep(0.2)
    out.write_bytes(orjson.dumps(all_items))
