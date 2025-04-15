import psycopg
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import logging
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("device_metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("device_metrics")

# Device configuration
type1_devices = {
    "HFLI001": {"deployed_at": "2025-1-17", "threshold": 0.39, "sensor": 6},
    "HFLT001": {"deployed_at": "2025-2-8", "threshold": 0.41, "sensor": 6},
    "HFLT002": {"deployed_at": "2025-2-8", "threshold": 0.41, "sensor": 6},
    "STMT002": {"deployed_at": "2025-2-12", "threshold": 0.5, "sensor": 1},
    "STMT003": {"deployed_at": "2025-1-9", "threshold": 0.23, "sensor": 6},
    "STMT004": {"deployed_at": "2025-1-12", "threshold": 0.15, "sensor": 5},
    "JKFL001": {"deployed_at": "2025-2-14", "threshold":0.10, "sensor": 6},
    "JKFL002": {"deployed_at": "2025-2-14", "threshold": 0.17, "sensor": 6},
    "JKFL003": {"deployed_at": "2025-1-7", "threshold": 0.41, "sensor": 6},
    "HFLV002": {"deployed_at": "2025-2-26", "threshold": 0.41, "sensor": 6},
    "HFLV001": {"deployed_at": "2025-2-26", "threshold": 0.41, "sensor": 6},
    "HFLV003": {"deployed_at": "2025-2-26", "threshold": 0.41, "sensor": 6},
}

# Database configuration
DB_CONFIG = {
    "dbname": "klvin_iot",
    "user": "klvin",
    "password": "K!v1n@1234",
    "host": "localhost",
    "port": "5432",
    "options": "-c search_path=sentinel"
}

def connect_to_db():
    """Establish database connection with error handling"""
    try:
        return psycopg.connect(**DB_CONFIG)
    except psycopg.Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def fetch_device_data(device_id, start_date, end_date):
    """
    Fetch device readings within the specified date range
    """
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT id, device_id, sensor_readings, time
                    FROM device_readings
                    WHERE device_id = %s
                    AND time BETWEEN %s AND %s
                    ORDER BY time;
                """
                
                cur.execute(query, [device_id, start_date, end_date])
                rows = cur.fetchall()

                if not rows:
                    logger.info(f"No data found for device {device_id} between {start_date} and {end_date}")
                    return pd.DataFrame()

                return pd.DataFrame(rows, columns=['id', 'device_id', 'sensor_readings', 'time'])
    except psycopg.Error as e:
        logger.error(f"Error fetching data for device {device_id}: {e}")
        return pd.DataFrame()

def extract_sensor_value_by_id(sensor_readings, sensor_id):
    """Extract sensor values from JSON by sensor_id"""
    try:
        if isinstance(sensor_readings, str):
            readings = json.loads(sensor_readings)
        else:
            readings = sensor_readings

        for reading in readings:
            if reading.get("sensor_id") == sensor_id:
                value = reading.get("value")
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
        return None
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        logger.error(f"Error extracting sensor value: {e}")
        return None

def process_vibration_data(data, sensor_id, positive_threshold, negative_threshold, window_minutes=15):
    """Process vibration data and calculate ON time in minutes"""
    if data.empty:
        return 0
    
    # Extract sensor values using the provided sensor_id
    sensor_values = []
    for idx, row in data.iterrows():
        value = extract_sensor_value_by_id(row["sensor_readings"], sensor_id)
        sensor_values.append(value)
    
    data['sensor_value'] = sensor_values
    
    # Convert time
    data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.tz_localize(None)
    
    # Classification
    data['ON'] = (data['sensor_value'] > positive_threshold) | (data['sensor_value'] < negative_threshold)
    
    # Calculate total ON time in seconds for this window
    total_on_seconds = 0
    
    if len(data) > 1 and any(data['ON']):
        # Sort by time
        data = data.sort_values('time')
        
        # Get start and end times
        start_time = data['time'].min()
        end_time = data['time'].max()
        
        # Calculate time differences between consecutive readings
        data['next_time'] = data['time'].shift(-1)
        data['time_diff'] = (data['next_time'] - data['time']).dt.total_seconds()
        
        # For the last row, set time_diff to 0
        data.loc[data.index[-1], 'time_diff'] = 0
        
        # Calculate ON time
        for idx, row in data.iterrows():
            if row['ON'] and not pd.isna(row['time_diff']):
                # If this reading is ON, count the time until the next reading
                total_on_seconds += row['time_diff']
    
    # Convert seconds to minutes (round to nearest minute)
    total_on_minutes = round(total_on_seconds / 60)
    
    return total_on_minutes

def update_machine_metrics(device_id, metric_type, metric_data):
    """Update the machine_metrics table with the new metrics"""
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                # Check if an entry for today already exists
                query = """
                    SELECT metric_id, metric_data
                    FROM machine_metrics
                    WHERE device_id = %s
                    AND metric_type = %s
                    AND metric_date = CURRENT_DATE;
                """
                
                cur.execute(query, [device_id, metric_type])
                existing = cur.fetchone()
                
                if existing:
                    # Update existing entry
                    query = """
                        UPDATE machine_metrics
                        SET metric_data = %s, created_at = NOW()
                        WHERE metric_id = %s;
                    """
                    cur.execute(query, [existing[1] + metric_data, existing[0]])
                    logger.info(f"Updated metrics for device {device_id}, type {metric_type}: +{metric_data} minutes")
                else:
                    # Insert new entry
                    query = """
                        INSERT INTO machine_metrics
                        (device_id, metric_type, metric_data, metric_date)
                        VALUES (%s, %s, %s, CURRENT_DATE);
                    """
                    cur.execute(query, [device_id, metric_type, metric_data])
                    logger.info(f"Inserted new metrics for device {device_id}, type {metric_type}: {metric_data} minutes")
                
                conn.commit()
    except psycopg.Error as e:
        logger.error(f"Error updating machine metrics for device {device_id}: {e}")

def process_device(device_id, device_config):
    """Process a single device and update metrics"""
    logger.info(f"Processing device: {device_id}")
    
    # Skip offline devices
    if device_config["threshold"] == "OFFLINE":
        logger.info(f"Device {device_id} is marked as OFFLINE, skipping")
        return
    
    try:
        # Convert threshold to float
        threshold = float(device_config["threshold"])
    except (ValueError, TypeError):
        logger.error(f"Invalid threshold for device {device_id}: {device_config['threshold']}")
        return
    
    # Get deployed date
    try:
        deployed_date = datetime.strptime(device_config["deployed_at"], "%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid deployment date for device {device_id}: {device_config['deployed_at']}")
        return
    
    # Calculate time window for this run
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=15)
    
    # Skip if current time is before deployment date
    if end_time.date() < deployed_date.date():
        logger.info(f"Device {device_id} deployment date ({deployed_date.date()}) is in the future, skipping")
        return
    
    # Adjust start_time if it's before deployment date
    if start_time.date() < deployed_date.date():
        start_time = deployed_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Fetch data for the last 15 minutes
    data = fetch_device_data(device_id, start_time, end_time)
    
    if data.empty:
        logger.info(f"No data found for device {device_id} in the specified time window")
        return
    
    # Process vibration data
    sensor_id = device_config["sensor"]
    on_minutes = process_vibration_data(
        data,
        sensor_id,
        threshold,  # Positive threshold
        -threshold  # Negative threshold
    )
    
    if on_minutes > 0:
        # Update machine metrics table
        update_machine_metrics(device_id, "op_hours", on_minutes)
    else:
        logger.info(f"Device {device_id} was not ON during this period")

def process_all_devices():
    """Process all devices in the configuration"""
    logger.info("Starting device metrics processing job")
    
    for device_id, config in type1_devices.items():
        try:
            process_device(device_id, config)
        except Exception as e:
            logger.error(f"Error processing device {device_id}: {e}")
    
    logger.info("Completed device metrics processing job")

def main():
    """Main function to start the scheduler"""
    logger.info("Starting device metrics processing service")
    
    # Create scheduler
    scheduler = BackgroundScheduler()
    
    # Schedule the job to run every day 
    scheduler.add_job(process_all_devices, 'interval', minutes=1440)
    
    # Start the scheduler
    scheduler.start()
    
    logger.info("Scheduler started, processing will occur every 15 minutes")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler")
        scheduler.shutdown()
        logger.info("Scheduler shutdown complete")

if __name__ == "__main__":
    # Run once immediately on startup
    process_all_devices()
    
    # Then start the scheduler for subsequent runs
    main()
