"""Phase 9: Real weather service supporting OpenWeatherMap with live Open-Meteo fallback."""
from __future__ import annotations

import os
from typing import Any
import requests
from dotenv import load_dotenv

load_dotenv()


# WMO Weather interpretation codes (WW)
WMO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherService:
    """Retrieves live meteorological data and computes crop microclimate risks."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("WEATHER_API_KEY", "").strip()

    def get_weather(
        self,
        city: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        """Fetch real-time weather from OpenWeatherMap or live Open-Meteo fallback."""
        target_city = (city or "").strip()
        if not target_city and latitude is None and longitude is None:
            # Default to Pune, Maharashtra (major agricultural hub)
            target_city = "Pune"

        # Try OpenWeatherMap if key is provided
        if self.api_key:
            try:
                owm_result = self._fetch_open_weather_map(target_city, latitude, longitude)
                if owm_result.get("success"):
                    return owm_result
            except Exception as e:
                print(f"OpenWeatherMap request failed: {e}. Falling back to Open-Meteo...")

        # Live Open-Meteo Fallback (No key needed, live satellite/model data)
        return self._fetch_open_meteo(target_city, latitude, longitude)

    def _fetch_open_weather_map(
        self,
        city: str,
        lat: float | None,
        lon: float | None,
    ) -> dict[str, Any]:
        if lat is None or lon is None:
            geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={self.api_key}"
            geo_resp = requests.get(geo_url, timeout=5)
            geo_data = geo_resp.json()
            if not geo_data:
                return {"success": False, "error": f"City '{city}' not found."}
            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]
            resolved_city = geo_data[0].get("name", city)
            country = geo_data[0].get("country", "")
        else:
            resolved_city = city or f"Coord({lat:.2f}, {lon:.2f})"
            country = ""

        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        resp = requests.get(weather_url, timeout=5)
        data = resp.json()

        if resp.status_code != 200:
            return {"success": False, "error": data.get("message", "OpenWeatherMap API error")}

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        condition = data["weather"][0]["description"].capitalize()
        wind_speed = data.get("wind", {}).get("speed", 0.0)
        rain_1h = data.get("rain", {}).get("1h", 0.0)

        risk_analysis = self._analyze_disease_risks(temp, humidity, rain_1h)

        return {
            "success": True,
            "provider": "OpenWeatherMap",
            "location": f"{resolved_city}, {country}".strip(", "),
            "latitude": lat,
            "longitude": lon,
            "temperature_c": round(temp, 1),
            "humidity_percentage": humidity,
            "condition": condition,
            "precipitation_mm": rain_1h,
            "wind_speed_kmh": round(wind_speed * 3.6, 1),
            "risk_analysis": risk_analysis,
        }

    def _fetch_open_meteo(
        self,
        city: str,
        lat: float | None,
        lon: float | None,
    ) -> dict[str, Any]:
        try:
            resolved_city = city
            country = ""
            if lat is None or lon is None:
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
                geo_resp = requests.get(geo_url, timeout=6)
                geo_data = geo_resp.json()
                if not geo_data.get("results"):
                    # Default coordinates for Pune, Maharashtra if geocoding fails
                    lat, lon = 18.5204, 73.8567
                    resolved_city = city or "Pune"
                    country = "India"
                else:
                    result = geo_data["results"][0]
                    lat = result["latitude"]
                    lon = result["longitude"]
                    resolved_city = result.get("name", city)
                    country = result.get("country", "")

            meteo_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&current="
                f"temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
                f"&timezone=auto"
            )
            meteo_resp = requests.get(meteo_url, timeout=6)
            data = meteo_resp.json()

            current = data.get("current", {})
            temp = current.get("temperature_2m", 25.0)
            humidity = current.get("relative_humidity_2m", 60)
            precip = current.get("precipitation", 0.0)
            weather_code = current.get("weather_code", 0)
            wind_speed = current.get("wind_speed_10m", 0.0)

            condition = WMO_CODE_MAP.get(weather_code, "Partly cloudy")
            risk_analysis = self._analyze_disease_risks(temp, humidity, precip)

            return {
                "success": True,
                "provider": "Open-Meteo Live API",
                "location": f"{resolved_city}, {country}".strip(", "),
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "temperature_c": round(temp, 1),
                "humidity_percentage": humidity,
                "condition": condition,
                "precipitation_mm": precip,
                "wind_speed_kmh": round(wind_speed, 1),
                "risk_analysis": risk_analysis,
            }
        except Exception as e:
            return {
                "success": False,
                "provider": "Unavailable",
                "location": city or "Local Farm",
                "error": f"Weather lookup failed: {str(e)}",
                "temperature_c": 25.0,
                "humidity_percentage": 65,
                "condition": "Seasonal normal",
                "precipitation_mm": 0.0,
                "wind_speed_kmh": 10.0,
                "risk_analysis": "Unable to contact live weather service. Use standard seasonal precautions.",
            }

    @staticmethod
    def _analyze_disease_risks(temperature: float, humidity: float, precipitation: float) -> str:
        """Compute microclimatic risk commentary for plant pathogens."""
        risks = []
        if humidity >= 80:
            if 15.0 <= temperature <= 22.0:
                risks.append(
                    "High risk for Late Blight (Phytophthora infestans) and downy molds due to cool, moist conditions."
                )
            elif 22.0 < temperature <= 30.0:
                risks.append(
                    "Favorable environment for Early Blight, Septoria leaf spot, and Bacterial spot development."
                )
            else:
                risks.append("Sustained high humidity creates potential for foliar fungal infections.")
        elif humidity < 45 and temperature > 28.0:
            risks.append(
                "Hot, dry conditions accelerate two-spotted spider mite reproduction and whitefly feeding activity."
            )

        if precipitation > 0.5:
            risks.append("Rainfall/leaf wetness facilitates bacterial leaf spot dissemination and soil splashing.")

        if not risks:
            risks.append("Current weather parameters are within normal seasonal range with moderate disease spread pressure.")

        return " ".join(risks)


# Default singleton
_default_weather = None


def get_weather_service() -> WeatherService:
    global _default_weather
    if _default_weather is None:
        _default_weather = WeatherService()
    return _default_weather
