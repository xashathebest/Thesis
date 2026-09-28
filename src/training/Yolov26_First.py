# ============================================================
# ORPHAN PROCESS - CAMPAIGN + ORDER TEMPLATE GENERATOR V1
# Tune + Minion Only
# ============================================================

!pip install openpyxl -q

import re
import pandas as pd
from google.colab import files
from datetime import datetime
import numpy as np
# ============================================================
# UPLOAD FILES
# ============================================================

print("Upload:")
print("1. Tune Active Campaigns.xlsx")
print("2. Minion Active Campaigns.xlsx")
print("3. FL_Campaign_CSV_Template.csv")

uploaded = files.upload()

# ============================================================
# IDENTIFY FILES
# ============================================================

tune_file = None
minion_file = None
template_file = None
exclusions_file = None
am_mapping_file = None

for filename in uploaded.keys():

    lower = filename.lower()

    if "tune_am_mapping" in lower:
        am_mapping_file = filename

    elif "tune" in lower:
        tune_file = filename

    elif "counterpart" in lower:
        exclusions_file = filename

    elif "minion" in lower:
        minion_file = filename

    elif "campaign_csv_template" in lower:
        template_file = filename

# ============================================================
# READ FILES
# ============================================================
# NOTE: this section was missing from the script as pasted to me.
# Since the Campaign CSV generated successfully in your run, this
# code must exist in your actual notebook -- paste it back in here
# if you'd like me to review it too (e.g. tune_df, minion_df,
# template_columns, today, tune_before all need to be defined
# before the filters below run). Left as a marker so this script
# fails loudly and early instead of hitting a confusing error deep
# in the Tune filter section if it's missing.
# ------------------------------------------------------------
# tune_df = pd.read_excel(tune_file)
# minion_df = pd.read_excel(minion_file, sheet_name=...)  # verify correct sheet
# template_columns = pd.read_csv(template_file).columns.tolist()
# today = datetime.now()
# tune_before = len(tune_df)
# ------------------------------------------------------------

tune_df = pd.read_excel(tune_file)

minion_df = pd.read_excel(minion_file)
am_df = pd.read_csv(am_mapping_file)

print("AM Mapping Columns:")
print(am_df.columns.tolist())
am_df["offer_id"] = (
    am_df["offer_id"]
    .astype(str)
    .str.strip()
)

am_lookup = (
    am_df
    .drop_duplicates("offer_id")
    .set_index("offer_id")["advertiser_manager"]
    .to_dict()
)
template_columns = pd.read_csv(
    template_file,
    nrows=0
).columns.tolist()

# ============================================================
# LOAD COUNTERPART EXCLUSIONS
# ============================================================

if exclusions_file:
    exclusions_df = pd.read_excel(exclusions_file)
    excluded_ids = set(
        exclusions_df["Identifier"].fillna("").astype(str).str.strip()
    )
    print("Excluded IDs:")
    print(sorted(excluded_ids))
else:
    excluded_ids = set()
# ============================================================
# DATE VARIABLES
# ============================================================

today = datetime.today()

date_filename = today.strftime("%m%d%Y")
date_display = today.strftime("%m/%d/%Y")

date_filename = today.strftime("%m%d%Y")
date_display = today.strftime("%m/%d/%Y")

# ============================================================
# TUNE FILTERS
# ============================================================

tune_before = len(tune_df)

tune_df["offer_id"] = (
    pd.to_numeric(tune_df["offer_id"], errors="coerce")
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.strip()
)

print("Before Tune counterpart exclusion:", len(tune_df))
tune_df = tune_df[~tune_df["offer_id"].isin(excluded_ids)]
print("After Tune counterpart exclusion:", len(tune_df))

# Exclude TSW / TEST / TESTING / ADFLOW / INTERNAL
tune_df = tune_df[
    ~tune_df["offer"]
    .fillna("")
    .str.lower()
    .str.contains(r"tsw|test|testing|adflow|internal", regex=True)
]

# Exclude Fluent Internal clients
tune_df = tune_df[
    ~tune_df["advertiser"].fillna("").str.lower().str.contains("fluent internal")
]

# Keep only blank or single-digit (0-9) offer_reference_id
def keep_tune_reference(value):
    if pd.isna(value):
        return True
    value = str(value).strip()
    if value == "":
        return True
    try:
        numeric = float(value)
        if numeric.is_integer() and 0 <= numeric <= 9:
            return True
    except ValueError:
        pass
    return False

tune_df = tune_df[tune_df["offer_reference_id"].apply(keep_tune_reference)]

# Exclude Pilot Program campaigns
tune_df = tune_df[~tune_df["offer"].fillna("").str.lower().str.contains("pilot program")]

# Remove rows missing critical data
tune_df = tune_df.dropna(subset=["advertiser", "offer_id", "offer"])
tune_df = tune_df[
    (tune_df["advertiser"].astype(str).str.strip() != "")
    & (tune_df["offer_id"].astype(str).str.strip() != "")
    & (tune_df["offer"].astype(str).str.strip() != "")
]

print(f"Tune removed: {tune_before - len(tune_df)} rows")
print("\nFINAL TUNE SURVIVORS")
print(tune_df[["offer_id", "offer"]])

# ============================================================
# MINION FILTERS
# ============================================================

minion_before = len(minion_df)

# Column J = Revenue
minion_revenue_col = minion_df.columns[9]
minion_df = minion_df[
    pd.to_numeric(minion_df[minion_revenue_col], errors="coerce").fillna(0) >= 1
]

# Exclude TSW / TEST / TESTING / INTERNAL / SYNDICATION
# (uses adGroupName because that's what becomes Campaign Name)
exclude_terms = r"tsw|test|testing|internal|syndication"
minion_df = minion_df[
    ~minion_df["adGroupName"].fillna("").str.lower().str.contains(exclude_terms, regex=True)
]

# Remove syndication campaigns (campaignName, separate field from adGroupName)
minion_df = minion_df[
    ~minion_df["campaignName"].fillna("").str.lower().str.contains("syndication")
]

# Exclude Fluent Internal clients
minion_df = minion_df[
    ~minion_df["advertiser"].fillna("").str.lower().str.contains("fluent internal")
]

# Keep only blank netsuiteLineId (Column F)
minion_df = minion_df[
    minion_df["netsuiteLineId"].fillna("").astype(str).str.strip().eq("")
]

# Exclude known Minion counterparts (identifier extracted from campaignName)
def extract_identifier(text):
    matches = re.findall(r"\b\d{4,6}\b", str(text))
    if matches:
        return matches[0]
    return ""

minion_df["counterpart_id"] = (
    minion_df["campaignName"]
    .fillna("")
    .apply(extract_identifier)
)

minion_df["adGroupId"] = (
    minion_df["adGroupId"]
    .astype(str)
    .str.strip()
)

print("Blocked by counterpart_id:")
print(
    minion_df[
        minion_df["counterpart_id"].isin(excluded_ids)
    ][["campaignName", "counterpart_id"]]
)

print("Blocked by adGroupId:")
print(
    minion_df[
        minion_df["adGroupId"].isin(excluded_ids)
    ][["campaignName", "adGroupId"]]
)

minion_df = minion_df[
    (~minion_df["counterpart_id"].isin(excluded_ids))
    &
    (~minion_df["adGroupId"].isin(excluded_ids))
]

# Remove rows missing critical data
minion_df = minion_df.dropna(subset=["advertiser", "adGroupId", "adGroupName"])
minion_df = minion_df[
    (minion_df["advertiser"].astype(str).str.strip() != "")
    & (minion_df["adGroupId"].astype(str).str.strip() != "")
    & (minion_df["adGroupName"].astype(str).str.strip() != "")
]

print(f"Minion removed: {minion_before - len(minion_df)} rows")
print("\nFINAL MINION SURVIVORS")
print(minion_df[["campaignName", "advertiser"]])

# ============================================================
# BUILD TUNE RECORDS
# ============================================================

tune_records = pd.DataFrame()

tune_records["Client"] = tune_df["advertiser"]
tune_records["CID"] = tune_df["offer_id"]
tune_records["Campaign Name"] = tune_df["offer"]

# ADD THESE HERE
tune_records["Account Manager"] = (
    tune_df["offer_id"]
    .astype(str)
    .str.strip()
    .map(am_lookup)
    .fillna("")
)

tune_records["Accrual Approver"] = (
    tune_records["Account Manager"]
)

tune_records["Account Executive (Campaign)"] = ""
tune_records["Campaign Type"] = np.where(
    tune_df["channel"].str.strip().eq("Engaged Install"),
    "Performance - Mobile Apps/Install/Engaged Install",
    "Performance - Drive to Site"
)

tune_records["Conversion Point"] = np.where(
    tune_df["channel"].str.strip().eq("Engaged Install"),
    "App Install",
    "Drive to Site Conversion"
)

# Split Tune verticals...
# Split Tune verticals, e.g. "Gaming: Casual Games"
# -> Vertical 1 = Gaming, Vertical 2 = Casual Games

if tune_df.empty:

    tune_records["Campaign Vertical 1"] = ""
    tune_records["Campaign Vertical 2"] = ""

else:

    vertical_split = (
        tune_df["vertical"]
        .fillna("")
        .astype(str)
        .str.split(":", n=1, expand=True)
    )

    if 0 in vertical_split.columns:
        tune_records["Campaign Vertical 1"] = (
            vertical_split[0]
            .fillna("")
            .str.strip()
        )
    else:
        tune_records["Campaign Vertical 1"] = ""

    if 1 in vertical_split.columns:
        tune_records["Campaign Vertical 2"] = (
            vertical_split[1]
            .fillna("")
            .str.strip()
        )
    else:
        tune_records["Campaign Vertical 2"] = ""

tune_records["Campaign Platform"] = "Tune"

tune_records["Campaign Details"] = (
    "DailyOrphan_Tune_" + date_display
)

# ============================================================
# BUILD MINION RECORDS
# ============================================================
# ============================================================
# REMOVE DUPLICATE MINION CAMPAIGNS
# ============================================================

minion_df = minion_df.drop_duplicates(
    subset=["adGroupId"]
)
print("Minion after dedupe:", len(minion_df))
# ============================================================
# BUILD MINION RECORDS
# ============================================================

minion_records = pd.DataFrame()
minion_records["Client"] = minion_df["advertiser"]
minion_records["CID"] = minion_df["adGroupId"]

minion_records["Campaign Name"] = minion_df["adGroupName"]
minion_records["Campaign Type"] = "Performance - Drive to Site Click"
minion_records["Conversion Point"] = "Drive to Site Click"
# NEW
minion_records["Account Manager"] = minion_df["accountManager"]
minion_records["Accrual Approver"] = minion_df["accountManager"]

# ============================================================
# OVERRIDE CERTAIN AMS
# ============================================================

replacement_map = {
    "Lexie Zorbas": "Katie McClanahan",
    "Bennett Wattles": "Katie McClanahan"
}

minion_records["Account Manager"] = (
    minion_records["Account Manager"]
    .replace(replacement_map)
)

minion_records["Accrual Approver"] = (
    minion_records["Accrual Approver"]
    .replace(replacement_map)
)

minion_records["Campaign Vertical 1"] = ""
minion_records["Campaign Vertical 2"] = ""

minion_records["Campaign Platform"] = "Minion"
minion_records["Campaign Details"] = (
    "DailyOrphan_Minion_" + date_display
)
# ============================================================
# COMBINE
# ============================================================

combined = pd.concat([tune_records, minion_records], ignore_index=True)
# ============================================================
# STANDARDIZE ACCOUNT MANAGERS
# ============================================================

replacement_map = {
    "Lexie Zorbas": "Katie McClanahan",
    "Bennett Wattles": "Katie McClanahan"
}

combined["Account Manager"] = (
    combined["Account Manager"]
    .replace(replacement_map)
)

combined["Accrual Approver"] = (
    combined["Accrual Approver"]
    .replace(replacement_map)
)
# External ID
combined["External ID"] = [f"C{i}{date_filename}" for i in range(1, len(combined) + 1)]

# Constants
combined["Subsidiary"] = "Fluent, Inc. : Fluent LLC"

combined["Account Executive (Campaign)"] = "House Account"

if "Account Manager" not in combined.columns:
    combined["Account Manager"] = ""

combined["Status"] = "3. Campaign Build"

if "Campaign Type" not in combined.columns:
    combined["Campaign Type"] = ""

if "Conversion Point" not in combined.columns:
    combined["Conversion Point"] = ""

combined["Initial Budget"] = 0

if "Accrual Approver" not in combined.columns:
    combined["Accrual Approver"] = ""

# ============================================================
# BUILD FINAL CAMPAIGN TEMPLATE
# ============================================================

final_df = pd.DataFrame(columns=template_columns)
for col in final_df.columns:
    final_df[col] = combined[col] if col in combined.columns else ""

# Final cleanup / validation
final_df = final_df[final_df["Client"].fillna("").astype(str).str.strip().ne("")]
final_df = final_df[final_df["CID"].fillna("").astype(str).str.strip().ne("")]
final_df = final_df[final_df["Campaign Name"].fillna("").astype(str).str.strip().ne("")]

# ============================================================
# RESULTS
# ============================================================

print("\n================================")
print("PROCESS COMPLETE")
print("================================")
print(f"Tune Included: {len(tune_records)}")
print(f"Minion Included: {len(minion_records)}")
print(f"Total Included: {len(final_df)}")

print("\nValidation")
print("Blank Clients:", final_df["Client"].isna().sum())
print("Blank CIDs:", final_df["CID"].isna().sum())
print("Blank Campaign Names:", final_df["Campaign Name"].isna().sum())

# ============================================================
# REPLACE BLANKS WITH N/A
# ============================================================

final_df = final_df.fillna("")

final_df = final_df.replace(
    r'^\s*$',
    'N/A',
    regex=True
)

print("Blank Campaign Names:", final_df["Campaign Name"].isna().sum())

# ============================================================
# EXPORT CAMPAIGN CSV
# ============================================================
fixes_df = pd.read_excel("Campaign_Fixes.xlsx")

for _, row in fixes_df.iterrows():
    field = row["Field"]
    original = row["Original"]
    replacement = row["Replacement"]

    if field in final_df.columns:
        final_df[field] = final_df[field].replace(
            original,
            replacement
        )



output_file = f"FL_Campaign_CSV_Template_{date_filename}.csv"

final_df.to_csv(
    output_file,
    index=False
)

print("\nGenerated:", output_file)
# ============================================================
# BUILD ORDER TEMPLATE
# ============================================================

campaign_output_df = final_df.copy()

order_df = pd.DataFrame()
order_df["External ID"] = campaign_output_df["External ID"]
order_df["External ID"] = (
    campaign_output_df["External ID"]
    .astype(str)
    .str.replace("^C", "IO", regex=True)
)
order_df["Client"] = campaign_output_df["Client"]
order_df["Sales Classification"] = "To Be Reviewed"

# NOTE: see flag above -- this currently concatenates External ID + Campaign
# Name. If "Campaign" is meant to hold the platform's CMP##### value
# (filled in manually after the Campaign CSV is uploaded), this line will
# need to change to a manual/blank field instead.
order_df["Campaign"] = (
    campaign_output_df["External ID"].astype(str)
    + " "
    + campaign_output_df["Campaign Name"].astype(str)
)

order_df["Campaign Type"] = campaign_output_df["Campaign Type"]
order_df["Rate Model"] = ""
order_df["BU + Department"] = "Corporate : Corporate"
order_df["Item"] = (
    campaign_output_df["Campaign Type"]
    .str.replace(" - ", " : ", regex=False)
)
order_df["Description"] = campaign_output_df["Campaign Name"]
order_df["Memo for Client"] = ""
order_df["DeliveryStepID"] = ""
order_df["CID/Tune Offer ID"] = campaign_output_df["CID"]
order_df["Campaign Details"] = campaign_output_df["Campaign Details"]
order_df["Rate"] = ""
order_df["Daily Cap"] = ""
order_df["Monthly Cap"] = ""
order_df["Max Scrub Rate"] = ""
order_df["Account Executive"] = "House Account"
order_df["Account Manager"] = campaign_output_df["Account Manager"]
order_df["Start Date"] = ""
order_df["End Date"] = ""
order_df["Capped/ Uncapped"] = "Capped"
order_df["Conversion Point"] = "Drive to Site Click"
order_df["Accrual Approval (Line)"] = campaign_output_df["Account Manager"]
order_df["Quantity"] = ""
order_df["Event"] = ""
order_df["Campaign Vertical 1"] = campaign_output_df["Campaign Vertical 1"]
order_df["Campaign Vertical 2"] = campaign_output_df["Campaign Vertical 2"]
order_df["Campaign Platform"] = campaign_output_df["Campaign Platform"]

order_df["Rate"] = pd.to_numeric(
    order_df["Rate"],
    errors="coerce"
)

order_df["Daily Cap"] = pd.to_numeric(
    order_df["Daily Cap"],
    errors="coerce"
)

order_df["Monthly Cap"] = 0
order_df["Max Scrub Rate"] = 0

mask = (
    order_df["Daily Cap"].notna()
    & order_df["Rate"].notna()
    & (order_df["Rate"] != 0)
)

order_df.loc[mask, "Quantity"] = (
    order_df.loc[mask, "Daily Cap"] * 30
    / order_df.loc[mask, "Rate"]
).round(0)

order_columns = [
    "External ID",
    "Client",
    "Sales Classification",
    "Campaign",
    "Campaign Type",
    "Rate Model",
    "BU + Department",
    "Item",
    "Description",
    "Memo for Client",
    "DeliveryStepID",
    "CID/Tune Offer ID",
    "Campaign Details",
    "Rate",
    "Daily Cap",
    "Monthly Cap",
    "Max Scrub Rate",
    "Account Executive",
    "Account Manager",
    "Start Date",
    "End Date",
    "Capped/ Uncapped",
    "Conversion Point",
    "Accrual Approval (Line)",
    "Quantity",
    "Event",
    "Campaign Vertical 1",
    "Campaign Vertical 2",
    "Campaign Platform",
]
order_df = order_df[order_columns]

# ============================================================
# EXPORT ORDER CSV
# ============================================================
# ============================================================
# EXPORT ORDER CSV
# ============================================================

# Fill blanks with N/A except these columns
leave_blank = [
    "Memo for Client",
    "DeliveryStepID",
    "Rate Model"
]

for col in order_df.columns:
    if col not in leave_blank:
        order_df[col] = (
            order_df[col]
            .fillna("")
            .replace(r'^\s*$', 'N/A', regex=True)
        )


# ============================================================
# EXPORT ORDER CSV
# ============================================================

order_output = f"FL_IO_CSV_Template_{date_filename}.csv"
order_df.to_csv(order_output, index=False)

print("\nGenerated:", order_output)

# ============================================================
# EXPORT ORDER XLSX + COLORS
# ============================================================

from openpyxl import load_workbook
from openpyxl.styles import PatternFill

xlsx_output = f"FL_IO_CSV_Template_{date_filename}.xlsx"

order_df.to_excel(
    xlsx_output,
    index=False
)

wb = load_workbook(xlsx_output)
ws = wb.active

yellow_fill = PatternFill(
    fill_type="solid",
    start_color="FFFF00",
    end_color="FFFF00"
)

blue_fill = PatternFill(
    fill_type="solid",
    start_color="ADD8E6",
    end_color="ADD8E6"
)

header_map = {}

for cell in ws[1]:
    header_map[cell.value] = cell.column
# Always highlight Campaign and Account Executive columns
# Add Excel formula to Quantity column

rate_col = header_map["Rate"]
daily_cap_col = header_map["Daily Cap"]
quantity_col = header_map["Quantity"]

from openpyxl.utils import get_column_letter

rate_letter = get_column_letter(rate_col)
daily_cap_letter = get_column_letter(daily_cap_col)

for row in range(2, ws.max_row + 1):

    formula = (
        f"=IFERROR(({daily_cap_letter}{row}*30)/"
        f"{rate_letter}{row},\"\")"
    )

    ws.cell(row=row, column=quantity_col).value = formula

yellow_columns = [
    "Campaign",
    "Account Executive",
    "Conversion Point",
    "Capped/ Uncapped",
    "Rate Model"
]

for col_name in yellow_columns:
    if col_name in header_map:

        col_num = header_map[col_name]

        for row in range(2, ws.max_row + 1):
            ws.cell(row=row, column=col_num).fill = yellow_fill

        for row in range(2, ws.max_row + 1):
            ws.cell(row=row, column=col_num).fill = yellow_fill
# Yellow highlight N/A cells
for row in ws.iter_rows(min_row=2):
    for cell in row:
        if str(cell.value).strip() == "N/A":
            cell.fill = yellow_fill

# Always highlight Campaign and Account Executive columns
yellow_columns = [
    "Campaign",
    "Account Executive",
    "Conversion Point"

]

for col_name in yellow_columns:
    if col_name in header_map:

        col_num = header_map[col_name]

        for row in range(2, ws.max_row + 1):
            ws.cell(row=row, column=col_num).fill = yellow_fill

# Blue highlight manual calculation columns
for col_name in ["Quantity", "Daily Cap", "Rate"]:
    if col_name in header_map:

        col_num = header_map[col_name]

        for row in range(2, ws.max_row + 1):

            cell = ws.cell(row=row, column=col_num)

            cell.fill = blue_fill

wb.save(xlsx_output)

print("Generated:", xlsx_output)

# ============================================================
# DOWNLOAD FILES
# ============================================================

files.download(output_file)
files.download(order_output)
files.download(xlsx_output)