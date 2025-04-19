import psycopg
import pandas as pd
import json
from datetime import datetime, timedelta

# Database configuration
DB_CONFIG = {
    "dbname": "klvin_iot",
    "user": "klvin",
    "password": "K!v1n@1234",
    "host": "localhost",
    "port": "5432",
    "options": "-c search_path=sentinel"
}

# Device configuration dictionary
TYPE1_DEVICES = {
    "HFLI001": {
        "deployed_at": "2025-1-17",
        "threshold": 0.39,
        "sensor": 6,
    },
    "HFLT001": {"deployed_at": "2025-2-8", "threshold": 0.41, "sensor": 6},
    "HFLT002": {"deployed_at": "2025-2-8", "threshold": 0.41, "sensor": 6},
    "STMT002": {
        "deployed_at": "2025-2-12",
        "threshold": "0.5",
        "sensor": 1,
    },
    "STMT003": {"deployed_at": "2025-1-9", "threshold": 0.23, "sensor": 6},
    "STMT004": {"deployed_at": "2025-1-12", "threshold": 0.15, "sensor": 5},
    "JKFL001": {
        "deployed_at": "2025-2-14",
        "threshold": "OFFLINE",
        "sensor": 6,
    },
    "JKFL002": {
        "deployed_at": "2025-2-14",
        "threshold": 0.17,
        "sensor": 6,
    },
    "JKFL003": {"deployed_at": "2025-1-7", "threshold": 0.41, "sensor": 6},
    "HFLV002": {"deployed_at": "2025-2-26", "threshold": 0.41, "sensor": 6},
    "HFLV001": {"deployed_at": "2025-2-26", "threshold": 0.41, "sensor": 6},
    "HFLV003": {"deployed_at": "2025-2-26", "threshold": 0.41, "sensor": 6},
}

# Map sensor number to column name
SENSOR_MAP = {
    1: "sX",
    2: "sY",
    3: "sZ",
    4: "t1",
    5: "t2",
    6: "sZ",  # Default to sZ (most common)
}

def connect_to_db():
    """Establish database connection with error handling"""
    try:
        return psycopg.connect(**DB_CONFIG)
    except psycopg.Error as e:
        print(f"Error connecting to database: {e}")
        raise

def fetch_device_data(device_id, start_date=None, end_date=None):
    """
    Fetch device readings with optional date range filtering
    """
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT id, device_id, sensor_readings, time
                    FROM device_readings
                    WHERE device_id = %s
                """
                params = [device_id]

                if start_date and end_date:
                    query += " AND time BETWEEN %s AND %s"
                    params.extend([start_date, end_date])

                query += " ORDER BY time;"

                cur.execute(query, params)
                rows = cur.fetchall()

                return pd.DataFrame(rows, columns=['id', 'device_id', 'sensor_readings', 'time'])
    except psycopg.Error as e:
        print(f"Error fetching data: {e}")
        raise

def extract_sensor_value(sensor_readings, sensor_type):
    """Extract sensor values from JSON with improved error handling"""
    try:
        if isinstance(sensor_readings, str):
            readings = json.loads(sensor_readings)
        else:
            readings = sensor_readings

        for reading in readings:
            if reading["sensor_type"] == sensor_type:
                value = reading["value"]
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
        return None
    except (json.JSONDecodeError, AttributeError, KeyError):
        return None

def process_vibration_data(data, sensor_column="sZ", positive_threshold=0.39, negative_threshold=-0.39, 
                          window_size=15, max_vibration=3.0, consecutive_off_count=100):
    """
    Process vibration data to calculate both ON/OFF times and count ON/OFF cycles

    Parameters:
    - data: DataFrame with sensor_readings
    - sensor_column: Column to use for vibration analysis 
    - positive_threshold: Upper threshold for ON state detection
    - negative_threshold: Lower threshold for ON state detection
    - window_size: Window size in minutes for ON time calculation
    - max_vibration: Maximum valid vibration value (values above are considered anomalies)
    - consecutive_off_count: Number of consecutive readings below threshold to confirm OFF state

    Returns:
    - results: List of dictionaries with ON/OFF times and cycle counts
    - processed_data: DataFrame with extracted sensor values
    """
    print(f"Processing vibration data using {sensor_column} column with threshold {positive_threshold}...")
    
    # Extract sensor values
    for sensor in ["sX", "sY", "sZ", "t1", "t2", "IRT", "s"]:
        data[sensor] = data["sensor_readings"].apply(lambda x: extract_sensor_value(x, sensor))

    # Convert time and create date column
    data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.tz_localize(None)
    data['date'] = data['time'].dt.date

    # Fill NaN/None values with 0 to prevent comparison errors
    for col in ["sX", "sY", "sZ", "t1", "t2", "IRT", "s"]:
        data[col] = data[col].fillna(0)
    
    # Clean vibration values: set values outside acceptable range to 0
    vibration_columns = ["sX", "sY", "sZ"]
    for col in vibration_columns:
        # Count extreme values before replacement
        extreme_values = ((data[col] > 7) | (data[col] < -7)).sum()
        extreme_percent = (extreme_values / len(data)) * 100
        print(f"Column {col}: {extreme_values} extreme values ({extreme_percent:.2f}% of data)")

        # Replace extreme values with 0
        data.loc[(data[col] > 7) | (data[col] < -7), col] = 0

    # Classification for ON/OFF time calculation
    # Motor is ON when vibration exceeds thresholds in either direction
    data['ON'] = (data[sensor_column] > positive_threshold) | (data[sensor_column] < negative_threshold)

    results = []
    window_delta = pd.Timedelta(minutes=window_size)

    for date, group in data.groupby('date'):
        group = group.sort_values('time').reset_index(drop=True)

        # Part 1: Calculate total ON/OFF time
        total_on_seconds = 0
        i = 0
        while i < len(group):
            if group.iloc[i]['ON']:
                current_time = group.iloc[i]['time']
                window_end_time = current_time + window_delta

                # Find all points in the window
                window_data = group[
                    (group['time'] >= current_time) &
                    (group['time'] < window_end_time)
                ]

                if len(window_data) > 1:
                    time_diff = (window_data['time'].max() - window_data['time'].min()).total_seconds()
                    total_on_seconds += min(time_diff, window_delta.total_seconds())

                i += len(window_data)
            else:
                i += 1

        # Calculate hours and minutes
        total_seconds_in_day = 24 * 3600
        total_off_seconds = total_seconds_in_day - total_on_seconds

        on_hours, on_minutes = divmod(int(total_on_seconds // 60), 60)
        off_hours, off_minutes = divmod(int(total_off_seconds // 60), 60)

        # Part 2: Count ON/OFF cycles
        cycle_count = 0
        j = 0
        while j < len(group):
            # Search for ON state, ignoring values > max_vibration
            # Make sure we handle None/NaN values safely
            if (group.loc[j, sensor_column] is not None and 
                group.loc[j, sensor_column] > positive_threshold and 
                group.loc[j, sensor_column] <= max_vibration):
                
                cycle_count += 1  # Found an ON cycle

                # Slide forward until an OFF reading or value > max_vibration
                while (j < len(group) and 
                       group.loc[j, sensor_column] is not None and
                       group.loc[j, sensor_column] > positive_threshold and 
                       group.loc[j, sensor_column] <= max_vibration):
                    j += 1  # Move forward
                    if j >= len(group):
                        break

                # Check for OFF confirmation (consecutive readings below threshold or > max_vibration)
                off_counter = 0
                while j < len(group):
                    # Safely check if sensor value is below threshold or above max
                    if (group.loc[j, sensor_column] is None or
                        group.loc[j, sensor_column] <= positive_threshold or 
                        group.loc[j, sensor_column] > max_vibration):
                        off_counter += 1
                    else:
                        off_counter = 0  # Reset if a value between threshold and max_vibration is found

                    # If we confirm OFF state with enough consecutive readings, break the loop
                    if off_counter >= consecutive_off_count:
                        break

                    j += 1  # Move forward
                    if j >= len(group):
                        break
            else:
                j += 1  # Move to the next data point

        # Store complete results
        results.append({
            'date': date,
            'on_hours': on_hours,
            'on_minutes': on_minutes,
            'off_hours': off_hours,
            'off_minutes': off_minutes,
            'total_on_seconds': total_on_seconds,
            'cycle_count': cycle_count
        })

    return results, data

def insert_metrics_to_db(results, device_id):
    """
    Insert metrics data into machine_metrics table
    
    Parameters:
    - results: List of dictionaries with processed metrics data
    - device_id: The device ID for which data is being inserted
    """
    if not results:
        print(f"No results to insert for device {device_id}")
        return
        
    print(f"\nInserting metrics into database for device {device_id}...")
    
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                for result in results:
                    date = result['date']
                    # Convert on_hours and on_minutes to decimal hours
                    on_hours_decimal = result['on_hours'] + (result['on_minutes'] / 60)
                    cycle_count = result['cycle_count']
                    
                    # Insert operating hours metric
                    cur.execute("""
                        INSERT INTO machine_metrics 
                        (device_id, metric_type, metric_date, metric_data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (device_id, metric_type, metric_date) 
                        DO UPDATE SET metric_data = EXCLUDED.metric_data;
                    """, (device_id, 'op_hours', date, on_hours_decimal))
                    
                    # Insert operation cycles metric
                    cur.execute("""
                        INSERT INTO machine_metrics 
                        (device_id, metric_type, metric_date, metric_data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (device_id, metric_type, metric_date) 
                        DO UPDATE SET metric_data = EXCLUDED.metric_data;
                    """, (device_id, 'op_cycles', date, cycle_count))
            
            conn.commit()
        print(f"Successfully inserted {len(results) * 2} metric records for device {device_id}.")
        
    except psycopg.Error as e:
        print(f"Error inserting metrics into database for device {device_id}: {e}")

def process_device(device_id, device_config, start_date, end_date):
    """Process a single device with its configuration"""
    print(f"\n{'='*50}")
    print(f"Processing device: {device_id}")
    print(f"{'='*50}")
    
    # Check if device is marked as offline
    if device_config.get("threshold") == "OFFLINE":
        print(f"Device {device_id} is marked as OFFLINE. Skipping.")
        return
    
    # Convert threshold to float if it's a string
    try:
        threshold = float(device_config.get("threshold", 0.41))
    except (ValueError, TypeError):
        print(f"Invalid threshold for device {device_id}. Skipping.")
        return
    
    # Get sensor column based on sensor number
    sensor_num = device_config.get("sensor", 6)  # Default to 6 (sZ) if not specified
    sensor_column = SENSOR_MAP.get(sensor_num, "sZ")
    
    try:
        # Fetch data
        print(f"Fetching data for device {device_id}...")
        data = fetch_device_data(device_id, start_date, end_date)
        if data.empty:
            print(f"No data found for device {device_id} in the specified date range.")
            return

        print(f"Retrieved {len(data)} records.")

        # Process data with device-specific parameters
        results, processed_data = process_vibration_data(
            data,
            sensor_column=sensor_column,
            positive_threshold=threshold,
            negative_threshold=-threshold,  # Use negative of the threshold
            window_size=15,
            max_vibration=3.0,
            consecutive_off_count=100
        )

        # Print results
        if results:
            print("\nResults summary:")
            print("Date        | ON Time  | OFF Time | Cycles")
            print("-" * 50)
            for result in results:
                print(f"{result['date']} | {result['on_hours']:02d}:{result['on_minutes']:02d} | "
                      f"{result['off_hours']:02d}:{result['off_minutes']:02d} | "
                      f"{result['cycle_count']:5d}")
            
            # Insert metrics into database
            insert_metrics_to_db(results, device_id)
        else:
            print(f"No results generated for device {device_id}.")

    except Exception as e:
        print(f"An error occurred while processing device {device_id}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"Starting multi-device metrics processing at {datetime.now()}")
    
    # Set time range for last day
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    print(f"Processing data for date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Process each device
    for device_id, device_config in TYPE1_DEVICES.items():
        process_device(device_id, device_config, start_date, end_date)
    
    print(f"\nAll devices processed at {datetime.now()}")

if __name__ == "__main__":
    main()
