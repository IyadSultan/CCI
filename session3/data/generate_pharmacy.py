"""
Generate synthetic VistA pharmacy data for KHCC CCI training.
Uses MRNs from vista_patients.csv so tables can be JOINed.
"""

import pandas as pd
import random
import os
from datetime import datetime, timedelta

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
patients = pd.read_csv(os.path.join(SCRIPT_DIR, "vista_patients.csv"))
mrn_dob = list(zip(patients["MRN"].astype(str), patients["DOB"]))

DRUGS = [
    # (drug_name, orderable_item, dosage, units_per_dose, units, noun, route, schedule, verb, qty, days, arabic_dosage, arabic_instructions)
    ("TAMOXIFEN 20MG TAB", "TAMOXIFEN", "20", 1, "MG", "TABLET", "ORAL", "QDAY", "TAKE", 30, 30, "1 حبة", "تناول الدواء بعد الأكل."),
    ("ANASTROZOLE 1MG TAB", "ANASTROZOLE", "1", 1, "MG", "TABLET", "ORAL", "QDAY", "TAKE", 30, 30, "1 حبة", "تناول الدواء بعد الأكل أو على معدة فارغة."),
    ("LETROZOLE 2.5MG TAB", "LETROZOLE", "2.5", 1, "MG", "TABLET", "ORAL", "QDAY", "TAKE", 30, 30, "1 حبة", "تناول الدواء في نفس الوقت يوميا."),
    ("CAPECITABINE 500MG TAB", "CAPECITABINE", "1000", 2, "MG", "TABLET", "ORAL", "BID", "TAKE", 120, 14, "2 حبة", "تناول الدواء بعد الأكل مع كوب ماء كبير."),
    ("IMATINIB 400MG TAB", "IMATINIB", "400", 1, "MG", "TABLET", "ORAL", "QDAY", "TAKE", 30, 30, "1 حبة", "تناول الدواء مع الأكل وكوب ماء كبير."),
    ("SUNITINIB 50MG CAP", "SUNITINIB", "50", 1, "MG", "CAPSULE", "ORAL", "QDAY", "TAKE", 28, 28, "1 كبسولة", "تناول الدواء مع أو بدون الأكل."),
    ("ERLOTINIB 150MG TAB", "ERLOTINIB", "150", 1, "MG", "TABLET", "ORAL", "QDAY", "TAKE", 30, 30, "1 حبة", "تناول الدواء على معدة فارغة قبل الأكل بساعة."),
    ("ONDANSETRON 8MG TAB", "ONDANSETRON", "8", 1, "MG", "TABLET", "ORAL", "Q8H", "TAKE", 30, 10, "1 حبة", "تناول الدواء قبل العلاج الكيماوي بنصف ساعة."),
    ("DEXAMETHASONE 4MG TAB", "DEXAMETHASONE", "4", 1, "MG", "TABLET", "ORAL", "BID", "TAKE", 20, 10, "1 حبة", "تناول الدواء بعد الأكل."),
    ("METOCLOPRAMIDE 10MG TAB", "METOCLOPRAMIDE", "1 TABLET", None, None, None, "ORAL", "Q6H", "TAKE", 30, 30, "1 حبة", "تناول الدواء 30 دقيقة قبل الوجبات وقبل النوم."),
    ("ESOMEPRAZOLE 40MG TAB", "ESOMEPRAZOLE", "1 TABLET", None, None, None, "ORAL", "QDAY", "TAKE", 30, 30, "1 حبة", "تناول الدواء قبل تناول وجبة الإفطار بساعة."),
    ("MORPHINE SULFATE 10MG TAB", "MORPHINE", "10", 1, "MG", "TABLET", "ORAL", "Q4H PRN", "TAKE", 30, 10, "1 حبة", "تناول الدواء عند الحاجة للألم."),
    ("FENTANYL 25MCG/HR PATCH", "FENTANYL", "1 PATCH", None, None, None, "TOPICAL", "Q72H", "APPLY", 10, 30, "1 لصقة", "ضع اللصقة على الجلد وغيرها كل 3 أيام."),
    ("ENOXAPARIN 40MG INJ", "ENOXAPARIN", "40", 1, "MG", "SYRINGE", "SUBCUTANEOUS", "QDAY", "INJECT", 30, 30, "1 حقنة", "حقن تحت الجلد في البطن."),
    ("FILGRASTIM 300MCG INJ", "FILGRASTIM", "300", 1, "MCG", "SYRINGE", "SUBCUTANEOUS", "QDAY", "INJECT", 7, 7, "1 حقنة", "حقن تحت الجلد."),
    ("CALCIUM CARBONATE/VITAMIN D 650MG TAB", "CALCIUM CARBONATE/VITAMIN D", "1 TABLET", None, None, None, "ORAL", "BID", "TAKE", 60, 30, "1 حبة", ""),
    ("CELECOXIB 200MG CAP", "CELECOXIB", "200", 1, "MG", "CAPSULE", "ORAL", "BID", "TAKE", 60, 30, "1 كبسولة", "تناول الدواء بعد الأكل."),
    ("ALENDRONATE 70MG TAB", "ALENDRONATE", "70", 1, "MG", "TABLET", "ORAL", "WEEKLY", "TAKE", 4, 28, "1 حبة", "تناول الدواء على معدة فارغة مع كوب ماء كامل."),
    ("GABAPENTIN 300MG CAP", "GABAPENTIN", "300", 1, "MG", "CAPSULE", "ORAL", "TID", "TAKE", 90, 30, "1 كبسولة", "تناول الدواء مع أو بدون الأكل."),
    ("metFORMIN 500MG TAB", "metFORMIN", "500", 1, "MG", "TABLET", "ORAL", "QDAY", "TAKE", 30, 30, "1 حبة", "تناول الدواء بعد الأكل. تناول كميات كافية من الماء."),
    ("CEFUROXIME AXETIL 500MG TAB", "CEFUROXIME", "500", 1, "MG", "TABLET", "ORAL", "Q12H", "TAKE", 28, 14, "1 حبة", "تناول الدواء بعد الأكل. تناول كميات كافية من الماء."),
    ("DOXYCYCLINE 100MG CAP", "DOXYCYCLINE", "100", 1, "MG", "CAPSULE", "ORAL", "QDAY", "TAKE", 30, 30, "1 كبسولة", "تناول الدواء بعد الأكل مع كوب ماء كبير."),
    ("PARACETAMOL 500MG TAB", "PARACETAMOL", "500", 1, "MG", "TABLET", "ORAL", "Q6H PRN", "TAKE", 60, 30, "1 حبة", "تناول الدواء عند الحاجة. لا تتجاوز 8 حبات في اليوم."),
    ("FUSIDIC ACID 20MG/GM CREAM 15G", "FUSIDIC ACID", "A THIN LAYER", None, None, None, "TOPICAL", "BID", "APPLY", 2, 30, "طبقة رقيقة", ""),
    ("NICOTINE 25MG PATCH", "NICOTINE", "1 PATCH", None, None, None, "TOPICAL", "Q12H", "APPLY", 60, 30, "1 لصقة", "لمدة 16 ساعة و تنزع عند النوم."),
]

PROVIDERS = [
    "ABU HASSAN,FALAH ISSA FALAH", "ABDEEN,GHADEER IBRAHIM FAHMI",
    "HAWARI,FERAS IBRAHIM IRSHAID", "AL-MAJALI,ABDULJALIL EID ABDLNABI",
    "QASEM,NADIA FAISAL AHMED", "HADDAD,FARIS OMAR KHALIL",
    "AL-KHATIB,RANIA MOHAMMAD", "SALEH,AHMAD HUSSEIN ALI",
    "BSOUL,LINA RASHED NASSER", "TARAWNEH,ISSA MAHMOUD",
]

CLINICS = [
    "KHCC-3C-SURGERY", "KHCC-4C-HUSSAM AL-HARIRI", "KHCC-ICU-CRITICAL CARE",
    "BREAST ONC-GHADEER", "MEDICAL ONC-NADIA", "LUNG ONC-FERAS",
    "DERM-ABDELJALIL ALMAJALI", "PULMONARY-FERAS HAWARI",
    "PEDIATRIC ONC-AHMAD", "KHCC-2B-MEDICAL ONCOLOGY",
    "GI ONC-ISSA TARAWNEH", "HEMATOLOGY-LINA BSOUL",
]

PHARMACISTS = [
    "ALI,NOOR MOH'D MAH'D", "ESSA,MARYAM ZIAD TARIQ",
    "SHANNIES,HALA BASSAM SARKES", "SROJI,HALA MOTAZ MOHAMMED",
    "ABABNEH,SAMAR KHALID", "QUTAISHAT,REEM AMIN",
    "ALDAOUD,LANA FAISAL", "KHADER,NOUR MAHMOUD",
]

DIVISIONS = [
    "CENTRAL OP PHARM", "PEDIATRIC OP PHARM",
    "INPATIENT PHARMACY", "ONCOLOGY PHARM",
]

STATUSES = ["ACTIVE", "EXPIRED", "DISCONTINUED", "DISCONTINUED (EDIT)"]
STATUS_WEIGHTS = [0.15, 0.40, 0.30, 0.15]

ALL_COLUMNS = [
    "MRN", "DOB", "ISSUE_DATE", "PATIENT_STATUS", "PROVIDER", "CLINIC", "DRUG",
    "QTY", "DAYS_SUPPLY", "Number_OF_REFILLS", "OERR_SIG", "ORDER_CONVERTED",
    "COPIES", "MAIL_WINDOW_PARK", "ENTERED_BY", "IB_NUMBER", "UNIT_PRICE_OF_DRUG",
    "DIVISION", "LOGIN_DATE", "FILL_DATE", "DISPENSED_DATE", "EXPIRATION_DATE",
    "FILLING_PERSON", "FINISHING_PERSON", "FINISH_DATE_TIME",
    "PHARMACY_ORDERABLE_ITEM", "PLACER_ORDER_Number", "DRUG_ALLERGY_INGREDIENTS",
    "WAS_THE_PATIENT_COUNSELED", "STATUS", "LAST_DISPENSED_DATE", "TYPE_OF_RX",
    "POE_RX", "DOSAGE_ORDERED", "CHECKING_PHARMACIST", "DISPENSE_UNITS_PER_DOSE",
    "UNITS", "NOUN", "ROUTE", "SCHEDULE", "VERB", "HOSPITAL_COVERAGE", "NDC",
    "DRUG_ALLERGY_INDICATION", "COPAY_TRANSACTION_TYPE", "NEXT_POSSIBLE_FILL",
    "DELETION_COMMENTS", "PHARMACIST", "WAS_COUNSELING_UNDERSTOOD",
    "RELEASED_DATE_TIME", "MEDICATION_ROUTES", "VERIFYING_PHARMACIST",
    "RETURNED_TO_STOCK", "CANCEL_DATE", "LAST_DISPENSED_DATE_HOLDER", "DAW_CODE",
    "HOLD_REASON", "HOLD_DATE", "AFFECTED_MEDICATION", "NEW_DRUG",
    "PATIENT_INSTRUCTIONS", "DURATION", "PENDING_NEXT_POSSIBLE_FILLDATE",
    "DDSTATUS", "FORWARD_ORDER_Number", "BINGO_WAIT_TIME", "HOLD_COMMENTS",
    "COPAY_TYPE_AUDIT", "CONJUNCTION", "REMARKS", "OTHER_LANGUAGE_DOSAGE",
    "OTHER_PATIENT_INSTRUCTIONS", "DISCONTINUE_TYPE", "SIG1", "LABEL_DATE_TIME",
    "PROVIDER_COMMENTS", "PRIOR_FILL_DATE", "PREVIOUS_ORDER_Number", "TRADE_NAME",
    "LOT_Number", "DATE_OF_DEATH_HISTORY", "METHOD_OF_PICK_UP", "RX_NUMBER",
    "EPHARMACY_SUSPENSE_HOLD_DATE", "PHARMACY_INSTRUCTIONS", "REJECT_INFO",
    "PARTIAL_DATE", "RE_TRANSMIT_FLAG", "DATE_NDC_VALIDATED", "NDC_VALIDATED_BY",
    "BILLING_ELIGIBILITY_INDICATOR", "COPAY_EXCEEDING_CAP", "COPAY_ACTIVITY_LOG",
    "COSIGNING_PHYSICIAN", "ORIGINAL_QTY", "EXPANDED_PATIENT_INSTRUCTIONS",
    "SERVICE_CONNECTED", "MILITARY_SEXUAL_TRAUMA", "AGENT_ORANGE_EXPOSURE",
    "IONIZING_RADIATION_EXPOSURE", "SOUTHWEST_ASIA_CONDITIONS",
    "HEAD_AND_OR_NECK_CANCER", "COMBAT_VETERAN", "PROJ_112_SHAD",
    "EXTERNAL_PLACER_ORDER_NUMBER", "EXTERNAL_APPLICATION",
    "PFSS_ACCOUNT_REFERENCE", "PFSS_CHARGE_ID", "INDICATION_FOR_USE", "TPB_RX",
    "CLOZAPINE_DOSAGE_MG_DAY", "WBC_RESULTS", "DATE_OF_WBC_TEST", "ANC_RESULTS",
    "SIGNATURE_STATUS", "CMOP_EVENT", "LOT_EXP", "ICD_DIAGNOSIS", "REFILL_LOG",
    "manual_insertion_date", "manual_upsertion_date", "NUMBER",
]


def fmt_dt(dt):
    return dt.strftime("%Y-%m-%dT00:00:00.0000000")


def generate_records(n_records=500):
    rows = []
    rx_counter = 600000
    order_counter = 4620000
    number_counter = 968000

    for _ in range(n_records):
        mrn, dob_str = random.choice(mrn_dob)
        drug_info = random.choice(DRUGS)
        (drug_name, orderable, dosage, units_per_dose, units, noun,
         route, schedule, verb, qty, days, arabic_dosage, arabic_instr) = drug_info

        provider = random.choice(PROVIDERS)
        clinic = random.choice(CLINICS)
        pharmacist = random.choice(PHARMACISTS)
        entered_by = random.choice(PHARMACISTS + PROVIDERS[:4])
        division = random.choice(DIVISIONS)

        base_year = random.randint(2017, 2025)
        base_month = random.randint(1, 12)
        base_day = random.randint(1, 28)
        issue_date = datetime(base_year, base_month, base_day)
        fill_date = issue_date + timedelta(days=random.randint(0, 2))
        dispensed_date = fill_date
        login_date = issue_date
        finish_date = fill_date
        expiration_date = issue_date + timedelta(days=days + random.choice([0, 0, 335]))

        refills = random.choices([0, 1, 2, 3, 5], weights=[0.4, 0.15, 0.15, 0.15, 0.15])[0]
        status = random.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        counseled = random.choices(["YES", "NO"], weights=[0.85, 0.15])[0]
        order_converted = random.choice(["EXPIRATION TO CPRS", None, None])

        next_fill = None
        cancel_date = None
        last_disp_holder = None
        if status in ("DISCONTINUED", "DISCONTINUED (EDIT)"):
            cancel_dt = issue_date + timedelta(days=random.randint(14, 180))
            next_fill_dt = issue_date + timedelta(days=days + 1)
            next_fill = next_fill_dt.strftime("%m/%d/%Y")
            cancel_date = cancel_dt.strftime("%m/%d/%Y")
            last_disp_holder = fill_date.strftime("%m/%d/%Y")

        patient_instructions = None
        if "PRN" in schedule:
            patient_instructions = f"{verb} AS NEEDED FOR PAIN."
        elif "BEFORE MEALS" in arabic_instr or "قبل" in arabic_instr:
            patient_instructions = f"{verb} 30 MINUTES BEFORE MEALS AND AT BEDTIME."

        remarks = None
        if random.random() < 0.15:
            remarks = f"CPRS Order #{order_counter} Edited."
        if random.random() < 0.08:
            remarks = f"New Order Created by editing Rx # {rx_counter - random.randint(1, 100)}."

        row = {col: None for col in ALL_COLUMNS}

        row["MRN"] = mrn
        row["DOB"] = dob_str
        row["ISSUE_DATE"] = fmt_dt(issue_date)
        row["PATIENT_STATUS"] = "SC"
        row["PROVIDER"] = provider
        row["CLINIC"] = clinic
        row["DRUG"] = drug_name
        row["QTY"] = qty
        row["DAYS_SUPPLY"] = days
        row["Number_OF_REFILLS"] = refills
        row["OERR_SIG"] = "YES"
        row["ORDER_CONVERTED"] = order_converted
        row["COPIES"] = 1
        row["MAIL_WINDOW_PARK"] = "WINDOW"
        row["ENTERED_BY"] = entered_by
        row["DIVISION"] = division
        row["LOGIN_DATE"] = fmt_dt(login_date)
        row["FILL_DATE"] = fmt_dt(fill_date)
        row["DISPENSED_DATE"] = fmt_dt(dispensed_date)
        row["EXPIRATION_DATE"] = fmt_dt(expiration_date)
        row["FINISHING_PERSON"] = pharmacist
        row["FINISH_DATE_TIME"] = fmt_dt(finish_date)
        row["PHARMACY_ORDERABLE_ITEM"] = orderable
        row["PLACER_ORDER_Number"] = order_counter
        row["WAS_THE_PATIENT_COUNSELED"] = counseled
        row["STATUS"] = status
        row["LAST_DISPENSED_DATE"] = fmt_dt(fill_date)
        row["TYPE_OF_RX"] = 0
        row["POE_RX"] = "YES"
        row["DOSAGE_ORDERED"] = dosage
        row["DISPENSE_UNITS_PER_DOSE"] = units_per_dose
        row["UNITS"] = units
        row["NOUN"] = noun
        row["ROUTE"] = route
        row["SCHEDULE"] = schedule
        row["VERB"] = verb
        row["HOSPITAL_COVERAGE"] = 0
        row["WAS_COUNSELING_UNDERSTOOD"] = "YES" if counseled == "YES" else None
        row["NEXT_POSSIBLE_FILL"] = next_fill
        row["CANCEL_DATE"] = cancel_date
        row["LAST_DISPENSED_DATE_HOLDER"] = last_disp_holder
        row["PATIENT_INSTRUCTIONS"] = patient_instructions
        row["OTHER_LANGUAGE_DOSAGE"] = arabic_dosage
        row["OTHER_PATIENT_INSTRUCTIONS"] = arabic_instr if arabic_instr else None
        row["REMARKS"] = remarks
        row["RX_NUMBER"] = rx_counter
        row["NUMBER"] = number_counter

        rx_counter += 1
        order_counter += 1
        number_counter += 1

        rows.append(row)

    return pd.DataFrame(rows, columns=ALL_COLUMNS)


if __name__ == "__main__":
    df = generate_records(500)
    out_path = os.path.join(SCRIPT_DIR, "vista_pharmacy.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} pharmacy records -> {out_path}")
    print(f"Unique patients: {df['MRN'].nunique()}")
    print(f"Unique drugs: {df['DRUG'].nunique()}")
    print(f"Date range: {df['ISSUE_DATE'].min()} to {df['ISSUE_DATE'].max()}")
    print(f"Status distribution:\n{df['STATUS'].value_counts()}")
    print(f"\nColumns: {len(df.columns)}")
    print(f"Sample row (key fields):")
    sample = df.iloc[0]
    for col in ["MRN", "DRUG", "QTY", "DAYS_SUPPLY", "ROUTE", "SCHEDULE", "STATUS", "CLINIC"]:
        print(f"  {col}: {sample[col]}")
