import psycopg
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Database configuration
DB_CONFIG = {
    "dbname": "klvin_iot",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432",
    "options": "-c search_path=sentinel"
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

def process_vibration_data(data, positive_threshold=0.39, negative_threshold=-0.39, window_size=15,
                          max_vibration=3.0, consecutive_off_count=100):
    """
    Process vibration data to calculate both ON/OFF times and count ON/OFF cycles

    Parameters:
    - data: DataFrame with sensor_readings
    - positive_threshold: Upper threshold for ON state detection
    - negative_threshold: Lower threshold for ON state detection
    - window_size: Window size in minutes for ON time calculation
    - max_vibration: Maximum valid vibration value (values above are considered anomalies)
    - consecutive_off_count: Number of consecutive readings below threshold to confirm OFF state

    Returns:
    - results: List of dictionaries with ON/OFF times and cycle counts
    - processed_data: DataFrame with extracted sensor values
    """
    print("Processing vibration data...")
    # Extract sensor values
    for sensor in ["sX", "sY", "sZ", "t1", "t2", "IRT", "s"]:
        data[sensor] = data["sensor_readings"].apply(lambda x: extract_sensor_value(x, sensor))

    # Convert time and create date column
    data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.tz_localize(None)
    data['date'] = data['time'].dt.date

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
    data['ON'] = (data['sZ'] > positive_threshold) | (data['sZ'] < negative_threshold)

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
            if group.loc[j, 'sZ'] > positive_threshold and group.loc[j, 'sZ'] <= max_vibration:
                cycle_count += 1  # Found an ON cycle

                # Slide forward until an OFF reading or value > max_vibration
                while j < len(group) and group.loc[j, 'sZ'] > positive_threshold and group.loc[j, 'sZ'] <= max_vibration:
                    j += 1  # Move forward
                    if j >= len(group):
                        break

                # Check for OFF confirmation (consecutive readings below threshold or > max_vibration)
                off_counter = 0
                while j < len(group):
                    if group.loc[j, 'sZ'] <= positive_threshold or group.loc[j, 'sZ'] > max_vibration:
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

def plot_vibration_data(data, positive_threshold, negative_threshold, max_vibration=3.0):
    """Create visualization of vibration data with thresholds"""
    plt.figure(figsize=(15, 8))

    for date, day_data in data.groupby('date'):
        plt.plot(day_data['time'], day_data['sZ'],
                label=f'sZ Values ({date})', alpha=0.7)

    plt.axhline(y=positive_threshold, color='r', linestyle='--',
                label=f'Positive Threshold ({positive_threshold})')
    plt.axhline(y=negative_threshold, color='b', linestyle='--',
                label=f'Negative Threshold ({negative_threshold})')
    plt.axhline(y=max_vibration, color='g', linestyle=':',
                label=f'Max Vibration ({max_vibration})')

    plt.title('Vibration (sZ) Values with Thresholds')
    plt.xlabel('Time')
    plt.ylabel('sZ Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt

def plot_daily_metrics(results):
    """Create visualization of ON/OFF hours and cycle counts by day"""
    dates = [result['date'] for result in results]
    on_hours = [result['on_hours'] + result['on_minutes']/60 for result in results]
    cycle_counts = [result['cycle_count'] for result in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot ON hours
    ax1.bar(dates, on_hours, color='green', alpha=0.7)
    ax1.set_ylabel('ON Hours')
    ax1.set_title('Daily ON Hours')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plot Cycle Counts
    ax2.bar(dates, cycle_counts, color='blue', alpha=0.7)
    ax2.set_ylabel('Cycle Count')
    ax2.set_title('Daily ON/OFF Cycles')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

def insert_metrics_to_db(results, device_id):
    """
    Insert metrics data into machine_metrics table
    
    Parameters:
    - results: List of dictionaries with processed metrics data
    - device_id: The device ID for which data is being inserted
    """
    print("\nInserting metrics into database...")
    
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
        print(f"Successfully inserted {len(results) * 2} metric records.")
        
    except psycopg.Error as e:
        print(f"Error inserting metrics into database: {e}")
        raise

def main():
    # Configuration parameters
    DEVICE_ID = 'JKFL001'
    POSITIVE_THRESHOLD = 0.41
    NEGATIVE_THRESHOLD = -0.41
    WINDOW_SIZE = 15  # minutes
    MAX_VIBRATION = 3.0  # Maximum valid vibration value
    CONSECUTIVE_OFF_COUNT = 100  # Readings to confirm OFF state

    # Optional date range (set to None to get all data)
    start_date = datetime.now() - timedelta(days=6)  # Last 6 days
    end_date = datetime.now()

    try:
        # Fetch data
        print(f"Fetching data for device {DEVICE_ID}...")
        if start_date and end_date:
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        data = fetch_device_data(DEVICE_ID, start_date, end_date)
        if data.empty:
            print("No data found for the specified device and date range.")
            return

        print(f"Retrieved {len(data)} records.")

        # Process data
        results, processed_data = process_vibration_data(
            data,
            POSITIVE_THRESHOLD,
            NEGATIVE_THRESHOLD,
            WINDOW_SIZE,
            MAX_VIBRATION,
            CONSECUTIVE_OFF_COUNT
        )

        # Print results
        print("\nResults summary:")
        print("Date        | ON Time  | OFF Time | Cycles")
        print("-" * 50)
        for result in results:
            print(f"{result['date']} | {result['on_hours']:02d}:{result['on_minutes']:02d} | "
                  f"{result['off_hours']:02d}:{result['off_minutes']:02d} | "
                  f"{result['cycle_count']:5d}")
        
        # Insert metrics into database
        insert_metrics_to_db(results, DEVICE_ID)

        # Create and save plots
        print("\nGenerating visualizations...")

        # Vibration data plot
        vibration_plot = plot_vibration_data(
            processed_data,
            POSITIVE_THRESHOLD,
            NEGATIVE_THRESHOLD,
            MAX_VIBRATION
        )
        vibration_plot.savefig(f"{DEVICE_ID}_vibration_data.png")
        print(f"Vibration plot saved as {DEVICE_ID}_vibration_data.png")

        # Daily metrics plot
        metrics_plot = plot_daily_metrics(results)
        metrics_plot.savefig(f"{DEVICE_ID}_daily_metrics.png")
        print(f"Daily metrics plot saved as {DEVICE_ID}_daily_metrics.png")

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
