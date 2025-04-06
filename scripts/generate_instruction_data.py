import pandas as pd
import json
import random
from datetime import datetime, timedelta
import os
from itertools import product
from tqdm import tqdm

instruction = """Below is an excerpt from an event log used in process mining. Each row represents an event and contains an event ID, case ID, activity, and timestamp. The event log may contain one or multiple event log imperfection patterns which may affect the case ID, activity, and/or timestamp attribute.

Your task is to:
1. Diagnose the event log by identifying imperfection patterns. Clearly state which issues you detected and explain how you identified them.
2. Propose a correction strategy for each detected issue you will follow to mitigate the imperfections.
3. Repair the event log and output the corrected version in the same format as above.

Structure your response as follows:
<diagnosis>
[List the detected imperfections with explanations]
</diagnosis>

<mitigation>
[Describe the correction strategies]
</mitigation>

<log>
[Provide the repaired event log]
</log>"""

### SPLIT EVENT LOG INTO CHUNKS ###
def split_event_log(df, batch_size=100, soft_limit=150):
    """
    Splits event log into batches while ensuring full traces (cases) are not broken.
    Ensures each case appears at least once.
    """
    case_ids = df["case_id"].unique().tolist()
    random.shuffle(case_ids)  # Shuffle case IDs to ensure randomness

    batches = []
    current_batch = []
    current_size = 0

    for case_id in case_ids:
        case_events = df[df["case_id"] == case_id]
        case_size = len(case_events)

        if current_size + case_size > soft_limit and current_batch:
            batches.append(pd.concat(current_batch))
            current_batch = []
            current_size = 0

        current_batch.append(case_events)
        current_size += case_size

    if current_batch:
        batches.append(pd.concat(current_batch))

    return batches


def apply_random_imperfections(df, patterns, num_samples=2):
    """
    Applies the selected patterns in random combinations of error rates to the dataframe.
    Only non-None patterns are applied so that each valid pattern is injected once per combination.
    Returns a list of tuples: (erroneous_df, applied_patterns) for each combination.
    """
    # Define available error rates (10% to 100%)
    error_rates = list(range(10, 110, 10))
    
    # Filter out None patterns so only valid ones are used for combinations.
    valid_patterns = [p for p in patterns if p is not None]
    
    # Create the full list of all combinations of error rates for the valid patterns.
    all_combinations = list(product(error_rates, repeat=len(valid_patterns)))
    
    # Randomly sample up to num_samples combinations from the full list.
    sampled_combinations = random.sample(all_combinations, min(num_samples, len(all_combinations)))
    
    results = []
    
    for rates in sampled_combinations:
        erroneous_df = df.copy()
        applied_patterns = {}
        
        for pattern, error_rate in zip(valid_patterns, rates):
            erroneous_df, affected_event_ids = pattern(erroneous_df, error_rate)
            pattern_name = pattern.__name__.replace('inject_', '').replace('_', ' ')
            applied_patterns[pattern_name] = {
                'error_rate': error_rate,
                'affected_ids': affected_event_ids
            }
        
        results.append((erroneous_df, applied_patterns))
    
    return results


def inject_form_based_event_capture(df, error_rate):
    """Injects the form-based event capture pattern by assigning a shared timestamp to selected events, shuffling the dataset, and restoring order."""
    df = df.copy()  # Ensure we don't modify the original DataFrame
    affected_events = df.sample(frac=error_rate / 100).index  # Select X% of all events
    affected_event_ids = []
    
    for idx in affected_events:
        case_id = df.at[idx, "case_id"]
        case_events = df[df["case_id"] == case_id]

        if len(case_events) > 1:
            # Choose a random timestamp from another event in the same case
            shared_timestamp_event = case_events.sample(n=1).iloc[0]
            df.at[idx, "timestamp"] = shared_timestamp_event["timestamp"]
            affected_event_ids.append(df.at[idx, "event_id"])
            affected_event_ids.append(shared_timestamp_event["event_id"])

    # Shuffle the entire DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Restore order by case_id and timestamp
    df = df.sort_values(by=["case_id", "timestamp"]).reset_index(drop=True)

    return df, affected_event_ids



def inject_unanchored_event(df, error_rate):
    """Injects inconsistent timestamp formats."""
    formats = ["%Y-%m-%d %H:%M:%S %z", "%m/%d/%Y %H:%M %z", "%d.%m.%Y %H:%M:%S %z"]
    affected_events = df.sample(frac=error_rate / 100).index
    affected_event_ids = []

    for idx in affected_events:
        original_timestamp = df.at[idx, "timestamp"]
        
        # Ensure it's a proper datetime object
        if isinstance(original_timestamp, str):
            try:
                original_timestamp = pd.to_datetime(original_timestamp, utc=True)  # Convert with timezone awareness
            except Exception:
                continue  # Skip if conversion fails

        if isinstance(original_timestamp, pd.Timestamp):
            timezone_info = original_timestamp.tz  # Extract timezone

            # Format the timestamp into a new format (keeping timezone)
            new_format = random.choice(formats)
            new_timestamp_str = original_timestamp.strftime(new_format)

            # Convert back to datetime with the same timezone
            df.at[idx, "timestamp"] = new_timestamp_str
            affected_event_ids.append(df.at[idx, "event_id"])
                                      
    return df, affected_event_ids


def inject_collateral_events(df, error_rate):
    """DISCONTINUED."""
    affected_cases = df["case_id"].unique()
    num_affected = int(len(affected_cases) * (error_rate / 100))
    selected_cases = random.sample(list(affected_cases), num_affected)

    for case in selected_cases:
        case_events = df[df["case_id"] == case]
        num_duplicates = int(len(case_events) * (error_rate / 100))

        for _ in range(num_duplicates):
            duplicate_event = case_events.sample(n=1).copy()
            duplicate_event["timestamp"] += timedelta(seconds=random.randint(1, 5))
            df = pd.concat([df, duplicate_event])

    return df


def inject_elusive_case(df, error_rate):
    """Removes case IDs from some events."""
    affected_events = df.sample(frac=error_rate / 100).index
    df.loc[affected_events, "case_id"] = None  # Remove case IDs
    affected_event_ids = df.loc[affected_events, "event_id"].to_list()
    return df, affected_event_ids


def inject_polluted_labels(df, error_rate):
    """Adds random strings to activity labels."""
    affected_events = df.sample(frac=error_rate / 100).index
    affected_event_ids = []
    for idx in affected_events:
        df.at[idx, "activity"] += " - " + str(random.randint(1000, 9999))
        affected_event_ids.append(df.at[idx, "event_id"])
        
    return df, affected_event_ids


def inject_distorted_label(df, error_rate):
    """Swaps two letters in activity labels."""
    affected_events = df.sample(frac=error_rate / 100).index
    affected_event_ids = []
    for idx in affected_events:
        activity = df.at[idx, "activity"]
        if len(activity) > 3:
            i = random.randint(0, len(activity) - 2)
            activity = activity[:i] + activity[i+1] + activity[i] + activity[i+2:]
        df.at[idx, "activity"] = activity
        affected_event_ids.append(df.at[idx, "event_id"])

    return df, affected_event_ids

def descriptive_output(applied_patterns):
    """
    Write the diagnosis and mitigation part of the instruction dataset.
    """
    diagnosis = "<diagnosis>\n"
    mitigation = "<mitigation>\n"
    
    for pattern, details in applied_patterns.items():
        
        affected_ids = details['affected_ids']
        error_rate = str(details['error_rate']) + "%"
        
        diagnosis += f"Detected {pattern}: {len(affected_ids)} affected events which corresponds to an error rate of {error_rate}. "
        
        if pattern == "form based event capture":
            diagnosis += "This pattern was detected because some events have identical timestamps as other events in their case. "
            #diagnosis += f"The following events were affected: {' '.join(affected_ids)}.\n"
            mitigation += f"To mitigate the {pattern}, I will adjust the erroneous timestamps based on typical activity duration in correct data samples.\n"
        
        elif pattern == "unanchored event":
            diagnosis += "This pattern was detected due to inconsistencies in timestamp formatting, such as variations in date-month or month-date format. "
            #diagnosis += f"The following events were affected: {' '.join(affected_ids)}.\n"
            mitigation += f"To mitigate the {pattern}, I will standardize all timestamps to a uniform format.\n"
        
        elif pattern == "elusive case":
            diagnosis += "This pattern was detected because some events are missing case IDs. "
            #diagnosis += f"The following events were affected: {' '.join(affected_ids)}.\n"
            mitigation += f"To mitigate the {pattern}, I will assign missing case IDs based on time proximity and missing activities in known cases.\n"
        
        elif pattern == "polluted labels":
            diagnosis += "This pattern was detected because activity labels contain extraneous identifiers, making them inconsistent. "
            #diagnosis += f"The following events were affected: {' '.join(affected_ids)}.\n"
            mitigation += f"To mitigate the {pattern}, I will extract and retain the core activity names, removing unnecessary elements.\n"
        
        elif pattern == "distorted label":
            diagnosis += "This pattern was detected because activity labels contain minor spelling errors or variations. "
            #diagnosis += f"The following events were affected: {' '.join(affected_ids)}.\n"
            mitigation += f"To mitigate the {pattern}, I will standardize activity labels to their predominant correct form.\n"
        
    diagnosis += "</diagnosis>\n\n"
    mitigation += "</mitigation>\n\n"
    
    return diagnosis + mitigation


def dataframe_to_csv_string(df):
    """Converts a DataFrame to a CSV-like string (without saving as a file)."""
    return df.to_csv(index=False)


def save_to_json(original_df, erroneous_df, applied_patterns, filename):
    """Saves event logs in JSON format with CSV-like string representation."""
    output_string = descriptive_output(applied_patterns) + "<log>\n" + dataframe_to_csv_string(original_df) + "\n</log>"
    
    output_data = [{
        "instruction": instruction,
        "input": dataframe_to_csv_string(erroneous_df),
        "output": output_string,
    }]

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)
        
        
def generate_data(input_dir, output_dir):
    """Generates a full set of instruction data."""
    for file in os.listdir(input_dir):
        print("CURRENTLY AT", file)
        # use dictionary comprehension to make dict of dtypes
        dict_dtypes = {x : 'str'  for x in ['case_id', 'event_id', 'activity']}  
        df = pd.read_csv(input_dir + file, dtype=dict_dtypes)
        
        batches = split_event_log(df)
        
        # Define available imperfection patterns
        timestamp_patterns = [inject_form_based_event_capture, inject_unanchored_event]
        case_id_patterns = [inject_elusive_case]
        label_patterns = [inject_polluted_labels, inject_distorted_label]

        # Generate all valid combinations (each group contributes at most one pattern)
        all_valid_combinations = list(product(
            [None] + timestamp_patterns,  # None means this group is not selected
            [None] + case_id_patterns,
            [None] + label_patterns
        ))

        # Remove the (None, None, None) combination (no patterns applied)
        all_valid_combinations.remove((None, None, None))
        
        for i, batch in tqdm(enumerate(batches), total = len(batches)):

            j = 0
            for pattern in all_valid_combinations:
                results = apply_random_imperfections(batch, pattern, 2)
                for erroneous_batch, applied_patterns in results:
                    save_to_json(batch, erroneous_batch, applied_patterns, f"{output_dir}{file}_batch_{i}_combination_{j}.json")
                    j+=1
                    

if __name__ == '__main__':
    # Input paths
    TRAIN_DIR = "../data/train/"
    TEST_DIR = "../data/test/"

    # Output paths
    TRAIN_OUTPUT = "../data/instruction/train/"
    TEST_OUTPUT = "../data/instruction/test/"

    generate_data(TRAIN_DIR, TRAIN_OUTPUT)
    generate_data(TEST_DIR, TEST_OUTPUT)