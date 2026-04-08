import csv
import random
from datetime import datetime, timedelta

random.seed(42)

OUTPUT = r"C:\Users\USER\OneDrive - King Hussein Cancer Center\CCI\claude_notebookLM\session_3\data\vista_vitals.csv"

# 80 unique fake encrypted MRNs (14-digit numbers)
MRNS = [str(random.randint(10000000000000, 29999999999999)) for _ in range(80)]

# 30 fake Arabic-style names
FIRST_NAMES = ["MAHMOUD", "AHMAD", "MOHAMMAD", "OMAR", "KHALED", "SAMI", "TAREQ", "FADI",
               "RAMI", "NASSER", "SONDOS", "RANA", "HALA", "LINA", "DINA", "NOUR",
               "LAYLA", "FATIMA", "SARA", "REEM", "YAZAN", "HAMZA", "ISSA", "BASIL",
               "ZAID", "MALEK", "JOUD", "DANA", "ASEEL", "MAIS"]
MIDDLE_NAMES = ["RAED", "GHASSAN", "SAMIR", "WALID", "NABIL", "HASSAN", "KAMAL", "ADEL",
                "FAISAL", "JAMAL", "MUNIR", "SALIM", "MAZEN", "TAREK", "BILAL"]
LAST_NAMES = ["ALDOWERY", "ABUALSUOD", "ALQUDAH", "ALHAJJ", "ALTARAWNEH", "ALZOUBI",
              "BATAYNEH", "HABASHNEH", "OBEIDAT", "RAWASHDEH", "ALSMADI", "ALNAWAISEH",
              "ALSHORMAN", "ALKHATIB", "ALMOMANI", "ALJBOUR", "ALRASHDAN", "ALKHATEEB",
              "ALSAWALHA", "ALDMOUR", "ALQAISI", "ALAWNEH", "ALHUSBAN", "ALSALEH",
              "ALMASRI", "ABUHAMDA", "ALBDOUR", "ABUARAB", "ALSHBOUL", "ALOMARI"]

ENTERED_BY = []
for i in range(30):
    ln = LAST_NAMES[i]
    fn = FIRST_NAMES[i]
    mn = random.choice(MIDDLE_NAMES) + " " + random.choice(MIDDLE_NAMES)
    ENTERED_BY.append(f"{ln},{fn} {mn}")

LOCATIONS = [
    "KHCC-4C-HUSSAM AL-HARIRI",
    "KHCC-3B-ONCOLOGY",
    "KHCC-5A-BMT UNIT",
    "KHCC-2D-SURGICAL",
    "KHCC-ER-EMERGENCY",
    "KHCC-ICU-CRITICAL CARE",
    "KHCC-OPD-OUTPATIENT",
]

VITAL_TYPES = ["BLOOD PRESSURE", "PULSE", "PULSE OXIMETRY", "RESPIRATION", "TEMPERATURE"]

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 3, 31, 23, 59, 59)
DATE_RANGE_SECONDS = int((END_DATE - START_DATE).total_seconds())

def fmt_dt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ".0000000"

def gen_rate(vital_type, abnormal):
    if vital_type == "BLOOD PRESSURE":
        if abnormal:
            sys = random.choice([random.randint(90, 99), random.randint(160, 180)])
            dia = random.choice([random.randint(50, 59), random.randint(100, 110)])
        else:
            sys = random.randint(100, 155)
            dia = random.randint(60, 95)
        return f"{sys}/{dia}"
    elif vital_type == "PULSE":
        if abnormal:
            return str(random.choice([random.randint(55, 59), random.randint(110, 130)]))
        return str(random.randint(60, 105))
    elif vital_type == "PULSE OXIMETRY":
        if abnormal:
            return str(random.randint(88, 92))
        return str(random.randint(94, 100))
    elif vital_type == "RESPIRATION":
        if abnormal:
            return str(random.choice([random.randint(12, 13), random.randint(24, 28)]))
        return str(random.randint(14, 22))
    elif vital_type == "TEMPERATURE":
        if abnormal:
            temp = random.choice([round(random.uniform(96.5, 97.0), 1),
                                  round(random.uniform(100.5, 103.5), 1)])
        else:
            temp = round(random.uniform(97.2, 99.5), 1)
        return str(temp)

# Generate ~400 vital sets (400 * 5 = 2000 rows)
num_sets = 400
rows = []
number = 57737676

for _ in range(num_sets):
    mrn = random.choice(MRNS)
    location = random.choice(LOCATIONS)
    entered_by = random.choice(ENTERED_BY)
    taken_offset = random.randint(0, DATE_RANGE_SECONDS)
    dt_taken = START_DATE + timedelta(seconds=taken_offset)
    entry_delay = random.randint(0, 30 * 60)  # 0-30 min
    dt_entered = dt_taken + timedelta(seconds=entry_delay)
    abnormal = random.random() < 0.15

    for vt in VITAL_TYPES:
        rate = gen_rate(vt, abnormal)
        rows.append({
            "NUMBER": number,
            "MRN": mrn,
            "DATE_TIME_VITALS_TAKEN": fmt_dt(dt_taken),
            "VITAL_TYPE": vt,
            "DATE_TIME_VITALS_ENTERED": fmt_dt(dt_entered),
            "HOSPITAL_LOCATION": location,
            "ENTERED_BY": entered_by,
            "RATE": rate,
        })
        number += 1

# Sort by NUMBER (already sequential, but explicit)
rows.sort(key=lambda r: r["NUMBER"])

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "NUMBER", "MRN", "DATE_TIME_VITALS_TAKEN", "VITAL_TYPE",
        "DATE_TIME_VITALS_ENTERED", "HOSPITAL_LOCATION", "ENTERED_BY", "RATE"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows to {OUTPUT}")
