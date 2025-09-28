import requests
import re
from datetime import datetime
import pandas as pd
import numpy as np
import calendar

class DataFetcher:
    def __init__(self):
        pass
    
    def update_status(self, message):
        print(message)
    
    def get_last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]
    
    def generate_url(self, year, month, from_date, from_obs, to_date, to_obs, stnm="96509"):
        base_url = "https://weather.uwyo.edu/cgi-bin/sounding"
        query_params = (
            f"?region=seasia"
            f"&TYPE=TEXT%3ALIST"
            f"&YEAR={year}"
            f"&MONTH={month:02d}"
            f"&FROM={from_date:02d}{from_obs}"
            f"&TO={to_date:02d}{to_obs}"
            f"&STNM={stnm}"
        )
        return base_url + query_params
    
    def fetch_rason_data(self, url):
        try:
            self.update_status(f"Attempting to fetch data from URL: {url}")
            response = requests.get(url, verify=False)
            if response.status_code == 200:
                self.update_status("Data fetched successfully")
                return response.text
            else:
                self.update_status(f"Failed to fetch data, status code: {response.status_code}")
                return None
        except Exception as e:
            self.update_status(f"Error while fetching data: {e}")
            return None
    
    def fetch_with_retry(self, year, month, start_day, start_obs, end_day, end_obs):
        def try_fetch(url):
            self.update_status(f"Attempting to fetch data from URL: {url}")
            data = self.fetch_rason_data(url)
            if data:
                return data
            return None
        
        original_month = month
        original_year = year
        
        while True:
            for to_day in range(end_day, start_day - 1, -1):
                url = self.generate_url(year, month, start_day, start_obs, to_day, end_obs)
                data = try_fetch(url)
                if data:
                    return data, year, month
            
            for from_day in range(start_day, end_day + 1):
                url = self.generate_url(year, month, from_day, start_obs, end_day, end_obs)
                data = try_fetch(url)
                if data:
                    return data, year, month
            
            mid_day = (start_day + end_day) // 2
            from_day, to_day = start_day, end_day
            
            while from_day <= mid_day and to_day >= mid_day:
                url = self.generate_url(year, month, from_day, start_obs, to_day, end_obs)
                data = try_fetch(url)
                if data:
                    return data, year, month
                from_day += 1
                to_day -= 1
            
            self.update_status(f"Failed to fetch data for {year}-{month:02d}")
            month += 1
            if month > 12:
                year += 1
                month = 1
            start_day = 1
            end_day = self.get_last_day_of_month(year, month)
            
            if year > original_year or (year == original_year and month > original_month):
                self.update_status("Data not available for the specified range and subsequent months")
                return None, None, None
    
    def analyze_rason(self, data, start_day, end_day):
        rason = []
        observed_dates = set()
        year = None
        month = None
        month_name = None
        
        for line in data.split('\n'):
            match = re.search(r'Observations at (\d{2})Z (\d{2}) (\w+) (\d{4})', line)
            if match:
                current_time = match.group(1)
                current_date = int(match.group(2))
                month_name = match.group(3)
                year = int(match.group(4))
                month = datetime.strptime(month_name, '%b').month
                observed_dates.add((current_date, current_time))
                rason.append({
                    "Tanggal": f"{current_date:02d}",
                    "Jam": current_time,
                    "CAPE": None,
                    "KI": None,
                    "LI": None,
                    "SI": None,
                    "TT": None,
                })
            elif 'Convective Available Potential Energy' in line:
                if rason:
                    rason[-1]["CAPE"] = float(line.split(':')[1].strip())
            elif 'Lifted index' in line:
                if rason:
                    rason[-1]["LI"] = float(line.split(':')[1].strip())
            elif 'K index' in line:
                if rason:
                    rason[-1]["KI"] = float(line.split(':')[1].strip())
            elif 'Showalter index' in line:
                if rason:
                    rason[-1]["SI"] = float(line.split(':')[1].strip())
            elif 'Totals totals index' in line:
                if rason:
                    rason[-1]["TT"] = float(line.split(':')[1].strip())
        
        for day in range(start_day, end_day + 1):
            for hour in ["00", "12"]:
                if (day, hour) not in observed_dates:
                    rason.append({
                        "Tanggal": f"{day:02d}",
                        "Jam": hour,
                        "CAPE": None,
                        "KI": None,
                        "LI": None,
                        "SI": None,
                        "TT": None,
                    })
        
        rason.sort(key=lambda x: (x["Tanggal"], x["Jam"]))
        
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
            month_name = datetime.now().strftime('%b')
        
        return rason, year, month, month_name
