from datetime import date, datetime
from pytz import utc, timezone
import requests

def get_secret():
    puzzle_num = get_puzzle_num()
    request_url = f"https://semantoru.com/yesterday/{puzzle_num+1}"
    response = requests.get(request_url, timeout=5)
    if response.status_code == 200:
        return response.content
    else:
        return "Not found error."

def get_guess(word: str):
    puzzle_num = get_puzzle_num()
    request_url = f"https://semantoru.com/guess/{puzzle_num}/{word}"
    print(request_url)
    response = requests.get(request_url, timeout=5)
    print(response.status_code)
    if response.status_code == 200:
        rtn = response.json()
        print(rtn)
        if rtn['rank'] == '正解!':
            return rtn
        elif rtn['rank'] > 1000:
            rtn['rank'] = '?'
            return rtn
    else:
        return {"guess": word, 
                "sim": None,
                "rank": None}
    
def get_puzzle_num():
    fisrt_day = date(2023, 4, 2)
    return (utc.localize(datetime.utcnow()).astimezone(timezone('Asia/Tokyo')).date() - fisrt_day).days