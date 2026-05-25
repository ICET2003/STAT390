"""Download state-level weather features.

The output is one row per state, using the state capital as the representative
location. Open-Meteo is the default because it does not require an API key and
can retrieve historical weather. OpenWeatherMap current weather is also
supported for parity with the reference notebook pattern.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
DEFAULT_START_DATE = "2014-08-01"
DEFAULT_END_DATE = "2014-09-30"
DEFAULT_OUTPUT_PATH = Path("data/weather/state_weather.csv")
DEFAULT_METADATA_PATH = Path("results/weather_download_metadata.json")

STATE_CAPITALS = {
    "AL": ("Montgomery", 32.3777, -86.3000),
    "AK": ("Juneau", 58.3019, -134.4197),
    "AZ": ("Phoenix", 33.4484, -112.0740),
    "AR": ("Little Rock", 34.7465, -92.2896),
    "CA": ("Sacramento", 38.5816, -121.4944),
    "CO": ("Denver", 39.7392, -104.9903),
    "CT": ("Hartford", 41.7658, -72.6734),
    "DE": ("Dover", 39.1582, -75.5244),
    "FL": ("Tallahassee", 30.4383, -84.2807),
    "GA": ("Atlanta", 33.7490, -84.3880),
    "HI": ("Honolulu", 21.3099, -157.8581),
    "ID": ("Boise", 43.6150, -116.2023),
    "IL": ("Springfield", 39.7817, -89.6501),
    "IN": ("Indianapolis", 39.7684, -86.1581),
    "IA": ("Des Moines", 41.5868, -93.6250),
    "KS": ("Topeka", 39.0473, -95.6752),
    "KY": ("Frankfort", 38.2009, -84.8733),
    "LA": ("Baton Rouge", 30.4515, -91.1871),
    "ME": ("Augusta", 44.3106, -69.7795),
    "MD": ("Annapolis", 38.9784, -76.4922),
    "MA": ("Boston", 42.3601, -71.0589),
    "MI": ("Lansing", 42.7325, -84.5555),
    "MN": ("Saint Paul", 44.9537, -93.0900),
    "MS": ("Jackson", 32.2988, -90.1848),
    "MO": ("Jefferson City", 38.5767, -92.1735),
    "MT": ("Helena", 46.5891, -112.0391),
    "NE": ("Lincoln", 40.8136, -96.7026),
    "NV": ("Carson City", 39.1638, -119.7674),
    "NH": ("Concord", 43.2081, -71.5376),
    "NJ": ("Trenton", 40.2206, -74.7597),
    "NM": ("Santa Fe", 35.6870, -105.9378),
    "NY": ("Albany", 42.6526, -73.7562),
    "NC": ("Raleigh", 35.7796, -78.6382),
    "ND": ("Bismarck", 46.8083, -100.7837),
    "OH": ("Columbus", 39.9612, -82.9988),
    "OK": ("Oklahoma City", 35.4676, -97.5164),
    "OR": ("Salem", 44.9429, -123.0351),
    "PA": ("Harrisburg", 40.2732, -76.8867),
    "RI": ("Providence", 41.8240, -71.4128),
    "SC": ("Columbia", 34.0007, -81.0348),
    "SD": ("Pierre", 44.3683, -100.3510),
    "TN": ("Nashville", 36.1627, -86.7816),
    "TX": ("Austin", 30.2672, -97.7431),
    "UT": ("Salt Lake City", 40.7608, -111.8910),
    "VT": ("Montpelier", 44.2601, -72.5754),
    "VA": ("Richmond", 37.5407, -77.4360),
    "WA": ("Olympia", 47.0379, -122.9007),
    "WV": ("Charleston", 38.3498, -81.6326),
    "WI": ("Madison", 43.0731, -89.4012),
    "WY": ("Cheyenne", 41.1400, -104.8202),
    "DC": ("Washington", 38.9072, -77.0369),
}


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def kelvin_to_fahrenheit(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float((value - 273.15) * 9 / 5 + 32)


def seconds_to_hours(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value / 3600)


def mean(values: list[float | int | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def total(values: list[float | int | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean))


def fetch_state_weather(
    state_code: str,
    city: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
            ]
        ),
        "daily": ",".join(["sunshine_duration", "daylight_duration"]),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    payload = fetch_json(f"{ARCHIVE_API_URL}?{urlencode(params)}")
    hourly = payload.get("hourly", {})
    daily = payload.get("daily", {})

    sunshine_seconds = mean(daily.get("sunshine_duration", []))
    daylight_seconds = mean(daily.get("daylight_duration", []))

    return {
        "state_code": state_code,
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "temperature_f": mean(hourly.get("temperature_2m", [])),
        "humidity": mean(hourly.get("relative_humidity_2m", [])),
        "precipitation": total(hourly.get("precipitation", [])),
        "wind_speed": mean(hourly.get("wind_speed_10m", [])),
        "sunlight_hours": None
        if sunshine_seconds is None
        else sunshine_seconds / 3600,
        "daylight_hours": None
        if daylight_seconds is None
        else daylight_seconds / 3600,
    }


def fetch_openweather_state_weather(
    state_code: str,
    city: str,
    latitude: float,
    longitude: float,
    api_key: str,
) -> dict:
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
    }
    payload = fetch_json(f"{OPENWEATHER_API_URL}?{urlencode(params)}")
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    clouds = payload.get("clouds", {})
    rain = payload.get("rain", {})
    snow = payload.get("snow", {})
    weather = payload.get("weather", [{}])
    sys = payload.get("sys", {})
    sunrise = sys.get("sunrise")
    sunset = sys.get("sunset")
    sunlight_hours = None
    if sunrise is not None and sunset is not None:
        sunlight_hours = seconds_to_hours(float(sunset) - float(sunrise))

    return {
        "state_code": state_code,
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_f": kelvin_to_fahrenheit(main.get("temp")),
        "feels_like_f": kelvin_to_fahrenheit(main.get("feels_like")),
        "temperature_min_f": kelvin_to_fahrenheit(main.get("temp_min")),
        "temperature_max_f": kelvin_to_fahrenheit(main.get("temp_max")),
        "humidity": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "visibility_m": payload.get("visibility"),
        "cloud_cover": clouds.get("all"),
        "wind_speed": wind.get("speed"),
        "wind_direction": wind.get("deg"),
        "wind_gust": wind.get("gust"),
        "rain_1h": rain.get("1h"),
        "rain_3h": rain.get("3h"),
        "snow_1h": snow.get("1h"),
        "snow_3h": snow.get("3h"),
        "sunlight_hours": sunlight_hours,
        "weather_main": weather[0].get("main") if weather else None,
        "weather_description": weather[0].get("description") if weather else None,
        "weather_code": weather[0].get("id") if weather else None,
        "timezone_offset_seconds": payload.get("timezone"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_weather(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    pause_seconds: float = 0.1,
) -> pd.DataFrame:
    rows = []
    for state_code, (city, latitude, longitude) in STATE_CAPITALS.items():
        rows.append(
            fetch_state_weather(
                state_code=state_code,
                city=city,
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
            )
        )
        time.sleep(pause_seconds)

    weather = pd.DataFrame(rows).sort_values("state_code")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output_path, index=False)

    metadata = {
        "source": "Open-Meteo Archive API",
        "api_url": ARCHIVE_API_URL,
        "api_key_required": False,
        "start_date": start_date,
        "end_date": end_date,
        "location_strategy": "US state capitals",
        "rows": int(len(weather)),
        "output_path": str(output_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return weather


def download_openweather_current(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    api_key: str | None = None,
    pause_seconds: float = 0.1,
) -> pd.DataFrame:
    api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenWeatherMap requires an API key. Set OPENWEATHER_API_KEY or pass "
            "--api-key. Use --provider open-meteo for the no-key historical source."
        )

    rows = []
    for state_code, (city, latitude, longitude) in STATE_CAPITALS.items():
        rows.append(
            fetch_openweather_state_weather(
                state_code=state_code,
                city=city,
                latitude=latitude,
                longitude=longitude,
                api_key=api_key,
            )
        )
        time.sleep(pause_seconds)

    weather = pd.DataFrame(rows).sort_values("state_code")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output_path, index=False)

    metadata = {
        "source": "OpenWeatherMap Current Weather API",
        "api_url": OPENWEATHER_API_URL,
        "api_key_required": True,
        "location_strategy": "US state capitals by latitude/longitude",
        "rows": int(len(weather)),
        "output_path": str(output_path),
        "note": "Current weather snapshot; not historical 2014 survey-period weather.",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return weather


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download state-capital weather data."
    )
    parser.add_argument(
        "--provider",
        choices=["open-meteo", "openweathermap"],
        default="open-meteo",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--api-key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.provider == "openweathermap":
        weather = download_openweather_current(
            output_path=args.output,
            metadata_path=args.metadata,
            api_key=args.api_key,
        )
    else:
        weather = download_weather(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output,
            metadata_path=args.metadata,
        )
    print(f"Wrote {len(weather)} state weather rows to {args.output}")
    print(f"Wrote download metadata to {args.metadata}")


if __name__ == "__main__":
    main()
