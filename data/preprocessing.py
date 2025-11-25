import re
import pandas as pd

file_path_healthy = "healthy_chest_ct_synthetic_radiology_reports.csv"
file_path_unhealthy = "postive_chest_ct_synthetic_radiology_reports.csv"

def split_impression_followup(report: str):
	"""Extract impression and follow-up from a single report text.

	Rules implemented:
	- Split the report by lines ("\n").
	- Find a line that is an impression header. Accepts formats like
	  "**IMPRESSION:**", "IMPRESSION:", case-insensitive, optionally bolded.
	- The impression block is taken as the consecutive non-empty lines after
	  the header (or the remainder of the header line if content follows the header on the same line),
	  stopping at the first blank line.
	- The first non-blank line after that blank line is taken as the follow-up.
	- The returned cleaned_report is the original report with the header, impression
	  block and the follow-up line removed.

	Returns: (impression:str, followup:str, cleaned_report:str)
	"""
	if not isinstance(report, str):
		return "", "", ""

	lines = report.splitlines()
	header_re = re.compile(r"^\s*\*{0,2}\s*IMPRESSION\s*\*{0,2}\s*: ?(.*)$", re.IGNORECASE)

	for i, line in enumerate(lines):
		m = header_re.match(line)
		if not m:
			continue

		after = m.group(1).strip()
		# Determine where the impression content starts
		if after:
			# Impression content begins on the same line as header
			impression_lines = [after]
			j = i + 1
		else:
			# Impression content starts on following lines
			impression_lines = []
			j = i + 1

		# Collect impression lines until a blank line or end
		while j < len(lines):
			if lines[j].strip() == "":
				break
			impression_lines.append(lines[j])
			j += 1

		# Find first non-blank line after the blank separator -> follow-up
		k = j
		while k < len(lines) and lines[k].strip() == "":
			k += 1

		followup = ""
		if k < len(lines):
			followup = lines[k].strip()

		# Build cleaned report: everything before header, and everything after followup
		before = lines[:i]
		after_followup_index = (k + 1) if (k < len(lines)) else j
		after_parts = lines[after_followup_index:]
		cleaned = "\n".join(before + after_parts).strip()

		impression = "\n".join([ln for ln in impression_lines]).strip()
		return impression, followup, cleaned

	# No impression header found
	return "", "", report


def extract_impression_followup(df, report_col: str = "Report"):
	"""Process a dataframe, extracting `Impression` and `FollowUp` columns.

	Adds columns: `Impression`, `FollowUp`, and updates `Report` to the cleaned text
	with the impression and follow-up removed.
	"""
	impressions = []
	followups = []
	cleaned_reports = []

	for idx, row in df.iterrows():
		rpt = row.get(report_col, "")
		imp, fup, cleaned = split_impression_followup(rpt)
		impressions.append(imp)
		followups.append(fup)
		cleaned_reports.append(cleaned)

	df = df.copy()
	df["Impression"] = impressions
	df["FollowUp"] = followups
	df[report_col] = cleaned_reports
	return df


# load reports
df_healthy = pd.read_csv(file_path_healthy)
df_unhealthy = pd.read_csv(file_path_unhealthy)

# Extract impression and follow-up into their own columns
df_healthy = extract_impression_followup(df_healthy, report_col="Report")
df_unhealthy = extract_impression_followup(df_unhealthy, report_col="Report")

# Save masked/processed versions (don't overwrite originals)
df_healthy.to_csv("healthy_chest_ct_synthetic_radiology_reports_masked.csv", index=False)
df_unhealthy.to_csv("postive_chest_ct_synthetic_radiology_reports_masked.csv", index=False)

if __name__ == "__main__":
	print("Processed and wrote masked CSV files for healthy and unhealthy datasets.")