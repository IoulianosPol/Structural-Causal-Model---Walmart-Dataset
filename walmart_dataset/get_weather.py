import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib


class HighPerformanceWeatherEnhancer:
    """
    Meteorological covariate augmentation system designed to systematically enrich
    transactional data. The architecture integrates:
    - Deterministic geospatial assignment mapping store typologies to metropolitan coordinates.
    - Asynchronous, batched API retrieval (via Open-Meteo) to minimize network latency.
    - An in-memory memoization (caching) layer to eliminate redundant network queries.
    - A stochastic imputation module to generate synthetic baseline approximations during API outages.
    """

    def __init__(self, max_workers=10):

        # Geospatial dictionary establishing the predefined candidate locations per store typology
        self.city_mapping = {
            'A': [
                {'city': 'New York', 'lat': 40.7128, 'lon': -74.0060},
                {'city': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
                {'city': 'Chicago', 'lat': 41.8781, 'lon': -87.6298}
            ],
            'B': [
                {'city': 'Houston', 'lat': 29.7604, 'lon': -95.3698},
                {'city': 'Phoenix', 'lat': 33.4484, 'lon': -112.0740},
                {'city': 'Philadelphia', 'lat': 39.9526, 'lon': -75.1652}
            ],
            'C': [
                {'city': 'Austin', 'lat': 30.2672, 'lon': -97.7431},
                {'city': 'Jacksonville', 'lat': 30.3322, 'lon': -81.6557},
                {'city': 'San Jose', 'lat': 37.3382, 'lon': -121.8863}
            ]
        }

        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

        # In-memory stateful caching mechanism to optimize data retrieval
        self.weather_cache = {}
        self.cache_lock = threading.Lock()

        # Concurrent execution parameters
        self.max_workers = max_workers

        # Thread-safe throttling mechanism to respect external API rate limits
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.1

        # Execution diagnostics and runtime metrics
        self.api_calls = 0
        self.cache_hits = 0

    def stable_hash(self, x):
        """
        Computes a deterministic MD5-based cryptographic hash.
        This ensures longitudinal consistency and reproducibility in spatial assignments across runs.
        """
        return int(hashlib.md5(str(x).encode()).hexdigest(), 16)

    def get_city_coordinates(self, store_type, store_id):
        """
        Deterministically allocates a geospatial coordinate pair to a specific retail unit
        based on its overarching typology and unique identifier.
        """
        cities = self.city_mapping[store_type]
        index = self.stable_hash(store_id) % len(cities)
        return cities[index]

    def _apply_rate_limit(self):
        """
        Enforces a temporal throttling heuristic to ensure compliance with external API request limits
        and prevent service denial.
        """
        with self.rate_limit_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()

    def map_weather_code(self, code):
        """
        Translates raw numerical World Meteorological Organization (WMO) encodings
        into standard categorical text variables for subsequent predictive modeling.
        """
        if pd.isna(code): return "Unknown"
        code = int(code)
        if code == 0:
            return "Clear"
        elif code in [1, 2, 3]:
            return "Clouds"
        elif code in [45, 48]:
            return "Fog"
        elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
            return "Rain"
        elif code in [71, 73, 75, 77, 85, 86]:
            return "Snow"
        elif code in [95, 96, 99]:
            return "Thunderstorm"
        else:
            return "Unknown"

    def get_weather_batch(self, date_city_pairs):
        """
        Executes aggregated, time-bounded meteorological data retrieval for distinct
        geospatial clusters. By coalescing temporal ranges, this method significantly
        reduces the absolute volume of outgoing HTTP requests.
        """
        if not date_city_pairs:
            return {}

        # Aggregate observation targets by geospatial coordinate keys
        grouped = {}
        for date, city in date_city_pairs:
            key = f"{city['lat']}_{city['lon']}"
            grouped.setdefault(key, {"city": city, "dates": set()})
            grouped[key]["dates"].add(date)

        results = {}

        for _, group in grouped.items():
            city = group["city"]
            dates = sorted(group["dates"])

            try:
                self._apply_rate_limit()

                # Define integration parameters for the external weather repository
                params = {
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "start_date": min(dates),
                    "end_date": max(dates),
                    "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                              "weathercode", "windspeed_10m_max", "relative_humidity_2m_mean"],
                    "timezone": "auto"
                }

                response = requests.get(self.base_url, params=params, timeout=30)
                if response.status_code != 200:
                    raise Exception("API failure")

                data = response.json()["daily"]
                cache_block = {}

                # Map temporal sequences to the retrieved operational covariates
                for i, date_str in enumerate(data["time"]):
                    raw_code = data["weathercode"][i]
                    text_condition = self.map_weather_code(raw_code)

                    cache_block[date_str] = {
                        "temperature_max": data["temperature_2m_max"][i],
                        "temperature_min": data["temperature_2m_min"][i],
                        "precipitation": data["precipitation_sum"][i],
                        "weather_condition": text_condition,
                        "wind_speed": data["windspeed_10m_max"][i],
                        "humidity": data["relative_humidity_2m_mean"][i],
                        "city": city["city"]
                    }

                # Securely update the thread-safe global cache
                with self.cache_lock:
                    for d, v in cache_block.items():
                        self.weather_cache[f"{city['city']}_{d}"] = v

                for d in dates:
                    results[f"{city['city']}_{d}"] = cache_block.get(d)

            except Exception as e:
                # Initiate stochastic imputation pipeline upon network failure
                for d in dates:
                    results[f"{city['city']}_{d}"] = self._synthetic_fallback(city, d)

        return results

    def _synthetic_fallback(self, city, date):
        """
        Implements a stochastic heuristic to generate synthetic, latitudinally and seasonally
        adjusted meteorological approximations. Acts as a robust failover mechanism during
        persistent external API unavailability.
        """
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
        except:
            dt = datetime.now()

        month = dt.month
        lat = city["lat"]

        # Determine baseline thermal parameters utilizing geospatial heuristics
        if month in [12, 1, 2]:
            base_temp = 2 if lat > 40 else 12
        elif month in [6, 7, 8]:
            base_temp = 26 if lat > 40 else 32
        else:
            base_temp = 18

        # Introduce stochastic variance (Gaussian noise) to approximate natural meteorological fluctuations
        base_temp += np.random.normal(0, 3)

        return {
            "temperature_max": base_temp + 3,
            "temperature_min": base_temp - 3,
            "precipitation": np.random.exponential(1),
            "weather_condition": "synthetic",
            "wind_speed": np.random.uniform(1, 15),
            "humidity": np.random.randint(40, 85),
            "city": city["city"],
        }

    def process_batch(self, batch):
        """
        Processes discrete data partitions, leveraging the in-memory caching layer
        to circumvent redundant API requests, subsequently queuing unresolved observations
        for batched network retrieval.
        """
        results = {}
        pending = []
        city_map = {}

        for idx, store_type, store_id, date in batch:
            # Standardize chronological indicators to string format for consistent cache key generation
            if hasattr(date, 'strftime'):
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = str(date)[:10]

            city = self.get_city_coordinates(store_type, store_id)
            city_map[idx] = city
            cache_key = f"{city['city']}_{date_str}"

            # Interrogate the local cache prior to remote data acquisition
            with self.cache_lock:
                if cache_key in self.weather_cache:
                    results[idx] = self.weather_cache[cache_key]
                else:
                    pending.append((date_str, city))

        # Execute remote acquisition for all cache misses
        if pending:
            fetched = self.get_weather_batch(pending)

            for idx in batch:
                i = idx[0]
                if i not in results:
                    city = city_map[i]
                    d = idx[3]
                    if hasattr(d, 'strftime'):
                        d_str = d.strftime("%Y-%m-%d")
                    else:
                        d_str = str(d)[:10]

                    key = f"{city['city']}_{d_str}"
                    results[i] = fetched.get(key)

        return results

    def add_weather_features(self, df, batch_size=1000):
        """
        Coordinates the parallel execution pipeline. Allocates processing threads to parse
        the primary DataFrame in discrete batches, ultimately returning the enriched feature matrix.
        """
        print(f"Processing dataset: {len(df)} rows")
        df = df.reset_index(drop=True)
        output = [None] * len(df)
        batch = []

        for i, row in df.iterrows():
            batch.append((i, row["Type"], row["Store"], row["Date"]))

        # Establish a concurrent execution context to parallelize remote data acquisition
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {}
            for i in range(0, len(batch), batch_size):
                f = ex.submit(self.process_batch, batch[i:i + batch_size])
                futures[f] = i

            # Await future resolution and map outputs to the target indices
            for f in as_completed(futures):
                res = f.result()
                for k, v in res.items():
                    output[k] = v

        weather_df = pd.DataFrame(output)

        # Horizontally concatenate the resultant covariate matrix with the primary dataset
        return pd.concat([df.reset_index(drop=True), weather_df], axis=1)


def get_city_and_weather(df):
    """
    Entry point interface initializing the meteorological augmentation framework.
    Instantiates the enhancer object and triggers the batch processing pipeline.
    """
    enhancer = HighPerformanceWeatherEnhancer(max_workers=15)
    return enhancer.add_weather_features(df, batch_size=2000)