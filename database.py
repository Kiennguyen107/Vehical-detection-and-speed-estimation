import psycopg2
from datetime import datetime

DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'database': 'traffic_violations',
    'user':     'postgres',
    'password': 1234,
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    """Tạo bảng nếu chưa có."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS speed_violations (
            id               SERIAL PRIMARY KEY,
            time             TEXT,
            date             TEXT,
            vehicle_id       INTEGER,
            speed            INTEGER,
            over_by          INTEGER,
            lane             TEXT,
            plate            TEXT,
            color            TEXT,
            confidence       REAL,
            status           TEXT,
            frame_path       TEXT,
            dist_violation   BOOLEAN DEFAULT FALSE
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS distance_violations (
            id           SERIAL PRIMARY KEY,
            time         TEXT,
            date         TEXT,
            id_behind    INTEGER,
            distance     REAL,
            safe_dist    REAL,
            speed        INTEGER,
            lane         TEXT,
            plate        TEXT DEFAULT 'Detecting...'
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_speed_violation(v: dict):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO speed_violations
        (time, date, vehicle_id, speed, over_by, lane, plate, color, confidence, status, frame_path, dist_violation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        v['Time'], v['Date'], v['ID'], v['Speed (km/h)'], v.get('Over By (km/h)', 0),
        v['Lane'], v['Plate'], v.get('Color', ''),
        v['Confidence (%)'], v['Status'],
        v.get('Frame Path', ''), v.get('Dist Violation', False)
    ))
    conn.commit()
    cur.close()
    conn.close()

def insert_distance_violation(dv: dict):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO distance_violations
        (time, date, id_behind, distance, safe_dist, speed, lane, plate)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        dv['Time'], dv['Date'],
        int(dv['ID_Behind']),
        float(dv['Distance (m)']),
        float(dv['Safe_Dist (m)']),
        int(dv['Speed (km/h)']) if dv['Speed (km/h)'] != "N/A" else None,
        dv['Lane'],
        dv.get('Plate', 'Detecting...')
    ))
    conn.commit()
    cur.close()
    conn.close()

def update_plate(vehicle_id: int, plate: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE speed_violations SET plate = %s WHERE vehicle_id = %s
    """, (plate, vehicle_id))
    conn.commit()
    cur.close()
    conn.close()

def update_color(vehicle_id: int, color: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE speed_violations SET color = %s WHERE vehicle_id = %s
    """, (color, vehicle_id))
    conn.commit()
    cur.close()
    conn.close()

def update_dist_violation(vehicle_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE speed_violations SET dist_violation = TRUE WHERE vehicle_id = %s
    """, (vehicle_id,))
    conn.commit()
    cur.close()
    conn.close()
    
def update_dist_plate(vehicle_id: int, plate: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE distance_violations SET plate = %s WHERE id_behind = %s
    """, (plate, vehicle_id))
    conn.commit()
    cur.close()
    conn.close()
